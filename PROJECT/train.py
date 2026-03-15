import torch
from model import HybridRumourModel
from data_loader import get_data_loaders
import torch.nn.functional as F

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
    model = HybridRumourModel(num_features=num_features, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_samples += len(data.y)

        train_acc = correct / total_samples
        print(f"Epoch {epoch+1:03d}: Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_samples += len(data.y)
    
    test_acc = correct / total_samples
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "hybrid_rumour_model.pth")
    print("Model saved to hybrid_rumour_model.pth")

if __name__ == "__main__":
    train()