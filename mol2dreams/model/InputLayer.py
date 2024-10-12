import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, global_mean_pool as gap
from mol2dreams.utils.layers import SKIPGAT

class InputLayer(nn.Module):
    def __init__(self):
        super(InputLayer, self).__init__()

    def forward(self, batch):
        raise NotImplementedError("InputLayer forward method not implemented")

# Input Layers
class CONV_GNN(InputLayer):
    def __init__(self, node_features, embedding_size_reduced):
        super(CONV_GNN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(node_features, embedding_size_reduced)
        self.reluinit = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv4 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, batch):
        x = batch.x.float()
        edge_index = batch.edge_index

        hidden = self.initial_conv(x, edge_index)
        hidden = self.reluinit(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = self.reluconv1(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = self.reluconv2(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.reluconv3(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = self.reluconv4(hidden)

        return hidden

class TRANSFORMER_CONV(InputLayer):
    def __init__(self, node_features, edge_features, embedding_size_reduced, heads_num=4, dropout=0.1):
        super(TRANSFORMER_CONV, self).__init__()
        torch.manual_seed(42)

        # Transformer Conv layers
        self.initial_conv = TransformerConv(node_features, embedding_size_reduced, heads=heads_num, beta=True,
                                            dropout=dropout, edge_dim=edge_features)
        self.reluinit = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                     beta=True, dropout=dropout, edge_dim=edge_features)
        self.reluconv1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                     beta=True, dropout=dropout, edge_dim=edge_features)
        self.reluconv2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                     beta=True, dropout=dropout)
        self.reluconv3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, concat=False,
                                     heads=heads_num, beta=True, dropout=dropout)
        self.reluconv4 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, batch):
        x = batch.x.float()
        edge_index = batch.edge_index
        edge_attribute = batch.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attribute)
        hidden = self.reluinit(hidden)

        hidden = self.conv1(hidden, edge_index, edge_attribute)
        hidden = self.reluconv1(hidden)
        hidden = self.conv2(hidden, edge_index, edge_attribute)
        hidden = self.reluconv2(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.reluconv3(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = self.reluconv4(hidden)

        return hidden

class GAT(InputLayer):
    def __init__(self, node_features, edge_features, embedding_size_reduced, heads_num=4, dropout=0.1):
        super(GAT, self).__init__()
        torch.manual_seed(42)

        # GAT layers
        self.initial_conv = GATConv(node_features, embedding_size_reduced, heads=heads_num, edge_dim=edge_features)
        self.skipgat1 = SKIPGAT(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                edge_features=edge_features, dropout_rate=dropout)
        self.skipgat2 = SKIPGAT(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                dropout_rate=dropout)
        self.skipgat3 = SKIPGAT(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                dropout_rate=dropout)
        self.skipgat4 = SKIPGAT(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                dropout_rate=dropout)
        self.mean_conv = GATConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num, concat=False)
        self.mean_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, batch):
        x = batch.x.float()
        edge_index = batch.edge_index
        edge_attribute = batch.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attribute)

        hidden = self.skipgat1(hidden, edge_index, edge_attribute=edge_attribute)
        hidden = self.skipgat2(hidden, edge_index)
        hidden = self.skipgat3(hidden, edge_index)
        hidden = self.skipgat4(hidden, edge_index)
        hidden = self.mean_conv(hidden, edge_index)

        return hidden

class NEIMS(InputLayer):
    def __init__(self):
        super(NEIMS, self).__init__()

    def forward(self, batch):
        return batch  # Pass-through for molecular fingerprints
