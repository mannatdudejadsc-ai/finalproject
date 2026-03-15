import torch
from model import HybridRumourModel, GATRumourModel, GGNNRumourModel
from data_loader import get_data_loaders
import torch.nn.functional as F

def train_model(model, train_loader, test_loader, device, epochs=20, lr=0.01):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):

        for data in train_loader:

            data = data.to(device)

            optimizer.zero_grad()

            out = model(data)

            loss = F.nll_loss(out, data.y)

            loss.backward()

            optimizer.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data in test_loader:

            data = data.to(device)

            out = model(data)

            pred = out.argmax(dim=1)

            correct += int((pred == data.y).sum())

            total += len(data.y)

    return correct / total

def train():
    # Hyperparameters
    num_features = 384
    hidden_dim = 64
    batch_size = 16
    lr = 0.01
    epochs = 20

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Model
    gat_model = GATRumourModel(num_features).to(device)
    ggnn_model = GGNNRumourModel(num_features).to(device)
    hybrid_model = HybridRumourModel(num_features).to(device)

    print("Training GAT baseline...")
    gat_acc = train_model(gat_model, train_loader, test_loader, device)

    print("Training GGNN baseline...")
    ggnn_acc = train_model(ggnn_model, train_loader, test_loader, device)

    print("Training Hybrid model...")
    hybrid_acc = train_model(hybrid_model, train_loader, test_loader, device)   

    import json

    results = {
        "GAT (Baseline)": gat_acc,
        "GGNN (Baseline)": ggnn_acc,
        "Hybrid (Ours)": hybrid_acc
    }

    with open("model_metrics.json", "w") as f:
        json.dump(results, f)

    print("Model comparison saved to model_metrics.json")

if __name__ == "__main__":
    train()
