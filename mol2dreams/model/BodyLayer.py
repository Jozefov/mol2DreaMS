import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, global_mean_pool as gap
from mol2dreams.utils.layers import SKIPblock

class BodyLayer(nn.Module):
    def __init__(self):
        super(BodyLayer, self).__init__()

    def forward(self, x, batch, mass_shift):
        raise NotImplementedError("Body forward method not implemented")

class CONV_BODY(BodyLayer):
    def __init__(self, embedding_size_gnn, embedding_size, output_size):
        super(CONV_BODY, self).__init__()
        torch.manual_seed(42)

        self.bottleneck = nn.Linear(embedding_size_gnn, embedding_size)

        self.skip1 = SKIPblock(embedding_size, embedding_size)
        self.skip2 = SKIPblock(embedding_size, embedding_size)
        self.skip3 = SKIPblock(embedding_size, embedding_size)
        self.skip4 = SKIPblock(embedding_size, embedding_size)
        self.skip5 = SKIPblock(embedding_size, embedding_size)
        self.skip6 = SKIPblock(embedding_size, embedding_size)
        self.skip7 = SKIPblock(embedding_size, embedding_size)
        self.relu_out_resnet = nn.ReLU()

        # For this example, the output layers are placeholders
        self.output_layer = nn.Linear(embedding_size, output_size)
        self.relu_out = nn.ReLU()

    def forward(self, x, batch, mass_shift):
        batch_index = batch.batch
        hidden = gap(x, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)

        hidden = self.relu_out_resnet(hidden)

        # Output layer
        out = self.output_layer(hidden)
        out = self.relu_out(out)

        out = out.type(torch.float64)
        return out

class TRANSFORMER_CONV_BODY(BodyLayer):
    def __init__(self, embedding_size_gnn, embedding_size, output_size):
        super(TRANSFORMER_CONV_BODY, self).__init__()
        torch.manual_seed(42)

        self.bottleneck = nn.Linear(embedding_size_gnn, embedding_size)

        self.skip1 = SKIPblock(embedding_size, embedding_size)
        self.skip2 = SKIPblock(embedding_size, embedding_size)
        self.skip3 = SKIPblock(embedding_size, embedding_size)
        self.skip4 = SKIPblock(embedding_size, embedding_size)
        self.skip5 = SKIPblock(embedding_size, embedding_size)
        self.skip6 = SKIPblock(embedding_size, embedding_size)
        self.skip7 = SKIPblock(embedding_size, embedding_size)
        self.relu_out_resnet = nn.ReLU()

        # For this example, the output layers are placeholders
        self.output_layer = nn.Linear(embedding_size, output_size)
        self.relu_out = nn.ReLU()

    def forward(self, x, batch, mass_shift):
        batch_index = batch.batch
        hidden = gap(x, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)

        hidden = self.relu_out_resnet(hidden)

        # Output layer
        out = self.output_layer(hidden)
        out = self.relu_out(out)

        out = out.type(torch.float64)
        return out

class BIDIRECTIONAL_BODY(BodyLayer):
    def __init__(self, embedding_size_gnn, embedding_size, output_size, use_graph=True):
        super(BIDIRECTIONAL_BODY, self).__init__()
        torch.manual_seed(42)

        self.use_graph = use_graph

        self.bottleneck = nn.Linear(embedding_size_gnn, embedding_size)

        self.skip1 = SKIPblock(embedding_size, embedding_size)
        self.skip2 = SKIPblock(embedding_size, embedding_size)
        self.skip3 = SKIPblock(embedding_size, embedding_size)
        self.skip4 = SKIPblock(embedding_size, embedding_size)
        self.skip5 = SKIPblock(embedding_size, embedding_size)
        self.skip6 = SKIPblock(embedding_size, embedding_size)
        self.skip7 = SKIPblock(embedding_size, embedding_size)
        self.relu_out_resnet = nn.ReLU()

        # For this example, the output layers are placeholders
        self.output_layer = nn.Linear(embedding_size, output_size)
        self.relu_out = nn.ReLU()

    def forward(self, x, batch, mass_shift):
        if self.use_graph:
            batch_index = batch.batch
            x = gap(x, batch_index)
        hidden = self.bottleneck(x)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)

        hidden = self.relu_out_resnet(hidden)

        # Output layer
        out = self.output_layer(hidden)
        out = self.relu_out(out)

        out = out.type(torch.float64)
        return out

class Regression_BODY(BodyLayer):
    def __init__(self, input_size, hidden_size, use_graph=True, dropout_rate=0.15):
        super(Regression_BODY, self).__init__()
        torch.manual_seed(42)

        self.use_graph = use_graph
        self.dropout_rate = dropout_rate

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.proj1 = nn.Linear(input_size, hidden_size)  # Projection layer for skip connection

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)  # Projection layer for skip connection

        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.proj3 = nn.Linear(hidden_size, hidden_size)  # Projection layer for skip connection

        self.fc7 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn7 = nn.BatchNorm1d(hidden_size // 2)
        self.fc8 = nn.Linear(hidden_size // 2, 1)  # Output is a single scalar

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, batch, mass_shift):
        if self.use_graph:
            batch_index = batch.batch
            x = gap(x, batch_index)

        # First layer
        identity = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        # Second layer with skip connection
        identity = self.proj1(identity)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x += identity  # Skip connection

        # Third and Fourth layers
        identity = x
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        # Fifth and Sixth layers with skip connection
        identity = self.proj2(identity)
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        x += identity  # Skip connection

        # Seventh layer and Output layer
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.dropout(x)
        out = self.fc8(x).type(torch.float64)

        return out