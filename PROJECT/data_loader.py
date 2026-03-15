import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sentence_transformers import SentenceTransformer

# Load MiniLM model (384 dimensional embeddings)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


class RumourDataset(Dataset):

    def __init__(self, csv_path="dataset/rumours.csv"):

        super().__init__()

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