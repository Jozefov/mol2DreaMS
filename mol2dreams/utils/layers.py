from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn

class SKIPGAT(nn.Module):
    def __init__(self, in_features, hidden_features, heads, edge_features=None, use_dropout=True, dropout_rate=0.1):
        super(SKIPGAT, self).__init__()

        self.relu1 = nn.ReLU()
        if use_dropout:
            self.conv1 = GATConv(in_features, hidden_features, heads=heads, dropout=dropout_rate, edge_dim=edge_features)
        else:
            self.conv1 = GATConv(in_features, hidden_features, heads=heads, edge_dim=edge_features)

        self.relu2 = nn.ReLU()
        if use_dropout:
            self.conv2 = GATConv(hidden_features * heads, int(in_features / heads), heads=heads, dropout=dropout_rate,
                                 edge_dim=edge_features)
        else:
            self.conv2 = GATConv(hidden_features * heads, int(in_features / heads), heads=heads,
                                 edge_dim=edge_features)

    def forward(self, x, edge_index, edge_attribute=None):
        hidden = self.relu1(x)
        hidden = self.conv1(hidden, edge_index, edge_attribute)

        hidden = self.relu2(hidden)
        hidden = self.conv2(hidden, edge_index, edge_attribute)
        hidden = hidden + x  # Skip connection

        return hidden

class SKIPblock(nn.Module):
    def __init__(self, in_features, hidden_features, bottleneck_factor=0.5, use_dropout=True, dropout_rate=0.2):
        super(SKIPblock, self).__init__()

        self.batchNorm1 = nn.BatchNorm1d(in_features)
        self.relu1 = nn.ReLU()
        if use_dropout:
            self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden1 = nn.utils.weight_norm(nn.Linear(in_features, int(hidden_features * bottleneck_factor)),
                                            name='weight', dim=0)

        self.batchNorm2 = nn.BatchNorm1d(int(hidden_features * bottleneck_factor))
        self.relu2 = nn.ReLU()
        if use_dropout:
            self.dropout2 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.utils.weight_norm(nn.Linear(int(hidden_features * bottleneck_factor), in_features),
                                            name='weight', dim=0)

    def forward(self, x):
        hidden = self.batchNorm1(x)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.hidden1(hidden)

        hidden = self.batchNorm2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.hidden2(hidden)

        hidden = hidden + x  # Skip connection

        return hidden