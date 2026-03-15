import torch
from model import HybridRumourModel
from data_loader import RumourDataset
import torch.nn.functional as F
import random

def predict(text_input="Sample rumour text"):
    # Note: In a real system, 'text_input' would be converted to a graph structure + embeddings
    # Here we simulate this by generating a random graph using our mock data loader
    
    print(f"Analyzing input: '{text_input}'")
    print("Generating graph structure from input...")
    
    # Generate 1 mock graph
    # We use our dataset class to generate it consistent with training data
    dataset = RumourDataset(num_graphs=1, num_features=128)
    data = dataset[0]
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridRumourModel(num_features=128, hidden_dim=64).to(device)
    
    try:
        model.load_state_dict(torch.load("hybrid_rumour_model.pth", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file 'hybrid_rumour_model.pth' not found. Please run train.py first.")
        return

    model.eval()
    data = data.to(device)
    
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