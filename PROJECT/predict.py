import torch
from model import HybridRumourModel
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def predict(text_input="Sample rumour text"):
    print(f"Analyzing input: '{text_input}'")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridRumourModel(num_features=384, hidden_dim=64).to(device)
    
    try:
        model.load_state_dict(torch.load("hybrid_rumour_model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: Model file 'hybrid_rumour_model.pth' not found. Please run train.py first.")
        return
        
    model.eval()
    
    # Preprocess text into graph format
    embeddings = embedder.encode([text_input.strip()])
    x = torch.tensor(embeddings, dtype=torch.float).to(device)
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    batch = torch.zeros(1, dtype=torch.long).to(device)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    with torch.no_grad():
        out = model(data)
        probs = torch.exp(out) # Convert log_softmax to probability
        pred_class = out.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()
    
    labels = ["Non-Rumour", "Rumour"]
    prediction = labels[pred_class]
    
    print("\n" + "="*30)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print("="*30 + "\n")
    
    return prediction, confidence

if __name__ == "__main__":
    text = input("Enter tweet/news text: ")
    predict(text)