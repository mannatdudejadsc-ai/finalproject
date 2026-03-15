import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    GatedGraphConv,
    GATConv,
    global_mean_pool
)

class HybridRumourModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super().__init__()

        # 1. Input projection (MANDATORY for GGNN)
        # GGNN requires input_dim == out_channels
        self.lin_in = Linear(num_features, hidden_dim)

        # 2. GGNN Layer (Structural Propagation)
        self.ggnn = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=3
        )

        # 3. GAT Layer (Attention)
        # concat=False keeps output dim = hidden_dim
        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False
        )

        # 4. Classifier
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Project input features
        x = self.lin_in(x)
        x = F.relu(x)

        # GGNN propagation
        x = self.ggnn(x, edge_index)
        x = F.relu(x)

        # GAT attention
        x = self.gat(x, edge_index)
        x = F.relu(x)

        # Global pooling (graph-level representation)
        x = global_mean_pool(x, batch)

        # Classification
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)