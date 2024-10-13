import torch
import torch.nn as nn
from mol2dreams.utils.layers import SKIPblock

class GlobalInputLayer(nn.Module):
    def __init__(self, input_size, output_size, num_skipblocks=1, hidden_size=None, bottleneck_factor=0.5, use_dropout=True, dropout_rate=0.2):
        """
        Initializes the GlobalInputLayer.

        Args:
            input_size (int): Number of input global features.
            output_size (int): Desired output size after processing.
            num_skipblocks (int): Number of SKIPblocks to apply.
            hidden_size (int, optional): Hidden layer size within SKIPblocks. Defaults to output_size * 2.
            bottleneck_factor (float, optional): Factor to reduce hidden features in SKIPblocks. Defaults to 0.5.
            use_dropout (bool, optional): Whether to use dropout in SKIPblocks. Defaults to True.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(GlobalInputLayer, self).__init__()

        hidden_size = hidden_size if hidden_size is not None else output_size * 2

        self.initial_linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.skipblocks = nn.ModuleList([
            SKIPblock(in_features=hidden_size, hidden_features=hidden_size, bottleneck_factor=bottleneck_factor, use_dropout=use_dropout, dropout_rate=dropout_rate)
            for _ in range(num_skipblocks)
        ])

        self.final_linear = nn.Linear(hidden_size, output_size)
        self.final_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, global_features):
        """
        Forward pass for GlobalInputLayer.

        Args:
            global_features (torch.Tensor): Tensor of shape [batch_size, input_size].

        Returns:
            torch.Tensor: Processed global features of shape [batch_size, output_size].
        """
        x = self.initial_linear(global_features)
        x = self.relu(x)

        for skipblock in self.skipblocks:
            x = skipblock(x)

        x = self.final_linear(x)
        x = self.final_relu(x)

        return x