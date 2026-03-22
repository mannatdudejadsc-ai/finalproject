import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sentence_transformers import SentenceTransformer

# Load MiniLM model (384 dimensional embeddings)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
ARTICLE_ID   = 6392078
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"
CSV_PATH     = "dataset/rumours.csv"

def download_pheme_if_needed():
    """
    Downloads PHEME-9 from Figshare and builds dataset/rumours.csv.
    Skipped entirely if the CSV already exists (won't re-download).
    """
    if os.path.exists(CSV_PATH):
        print(f"[PHEME] CSV already exists at {CSV_PATH}, skipping download.")
        return

    print("[PHEME] Fetching file list from Figshare API...")
    resp = requests.get(FIGSHARE_API)
    resp.raise_for_status()
    files = resp.json()

    zip_files = [f for f in files if f["name"].endswith(".zip")]
    main_file = max(zip_files, key=lambda f: f["size"])
    print(f"[PHEME] Downloading {main_file['name']} "
          f"({main_file['size'] // 1024 // 1024} MB)...")

    resp = requests.get(main_file["download_url"], timeout=300)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))

    rows = []
    for name in z.namelist():
        parts = name.strip("/").split("/")
        # Expected: event/rumours|non-rumours/thread_id/source-tweets/id.json
        #       or: event/rumours|non-rumours/thread_id/reactions/id.json
        if len(parts) < 5 or not parts[-1].endswith(".json"):
            continue

        event     = parts[0]
        label_str = parts[1]   # 'rumours' or 'non-rumours'
        thread_id = parts[2]
        folder    = parts[3]   # 'source-tweets' or 'reactions'
        label     = 0 if label_str == "non-rumours" else 1

        with z.open(name) as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        tweet_id = str(data.get("id_str", parts[-1].replace(".json", "")))
        text     = data.get("text", "")

        if folder in ("source-tweets", "source-tweet"):
            parent_id = tweet_id          # root node points to itself
        elif folder == "reactions":
            parent_id = str(
                data.get("in_reply_to_status_id_str") or thread_id
            )
        else:
            continue

        rows.append({
            "thread_id": thread_id,
            "tweet_id":  tweet_id,
            "parent_id": parent_id,
            "text":      text,
            "label":     label,
            "event":     event,
            "is_source": 1 if folder in ("source-tweets", "source-tweet") else 0
        })

    os.makedirs("dataset", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"[PHEME] Saved {CSV_PATH} — "
          f"{len(df)} rows across {df['thread_id'].nunique()} threads.")


class RumourDataset(Dataset):

    def __init__(self, csv_path="dataset/rumours.csv"):

        super().__init__()
        download_pheme_if_needed()

        self.df = pd.read_csv(csv_path)
        self.graphs = self.build_graphs()

    def text_embedding(self, text):

        embedding = embedder.encode(text)

        return torch.tensor(embedding, dtype=torch.float)

    def build_graphs(self):

        graphs = []

    # group tweets by thread
        for thread_id, group in self.df.groupby("thread_id"):

            node_features = []
            edges = []
            node_map = {}

        # create nodes
            for i, (_, row) in enumerate(group.iterrows()):
                node_map[row["tweet_id"]] = i
                node_features.append(self.text_embedding(row["text"]))

        # create edges
            for _, row in group.iterrows():

                parent = row["parent_id"]
                child = row["tweet_id"]

                if parent in node_map and parent != child:
                    edges.append([node_map[parent], node_map[child]])

        # convert node features
            x = torch.stack(node_features)

        # convert edges
            if len(edges) > 0:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.empty((2,0), dtype=torch.long)

        # label
            y = torch.tensor([group.iloc[0]["label"]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)

            graphs.append(data)

        return graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


def get_data_loaders(batch_size=32):

    dataset = RumourDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
