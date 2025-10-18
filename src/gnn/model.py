import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import GCNConv

class EdgeRegressionGNN(nn.Module):
    """
    Simple stacked GCN for edge attribute + node feature enriched predictions.
    After message passing, edge embeddings are constructed by concatenating incident node reps + original edge_attr.
    """
    def __init__(self, node_in: int, edge_in: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=1)
        out = self.edge_mlp(edge_repr)
        return out