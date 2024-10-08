import torch
import torch.nn as nn
from mol2dreams.model.HeadLayer import IdentityHead

class Mol2DreaMS(nn.Module):
    def __init__(self, input_layer: nn.Module, body_layer: nn.Module, head_layer: nn.Module = None):
        super(Mol2DreaMS, self).__init__()
        self.input_layer = input_layer
        self.body_layer = body_layer
        self.head_layer = head_layer if head_layer is not None else IdentityHead()

    def forward(self, batch):

        x = self.input_layer(batch)
        x = self.body_layer(x, batch)
        x = self.head_layer(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    