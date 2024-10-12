import torch
import torch.nn as nn

class HeadLayer(nn.Module):
    def __init__(self):
        super(HeadLayer, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Head forward method not implemented")

# Head (Placeholder for future use)
class IdentityHead(HeadLayer):
    def __init__(self):
        super(IdentityHead, self).__init__()

    def forward(self, x):
        return x  # Pass-through for now

class ProjectionHead(HeadLayer):
    def __init__(self, input_size, output_size):
        super(ProjectionHead, self).__init__()
        self.forward_prediction = nn.Linear(input_size, output_size)


    def forward(self, x):
        x = self.forward_prediction(x)

        return x

class BidirectionalHeadLayer(HeadLayer):
    def __init__(self, input_size, output_size):
        super(BidirectionalHeadLayer, self).__init__()
        torch.manual_seed(42)

        self.forward_prediction = nn.Linear(input_size, output_size)
        self.backward_prediction = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)



    def forward(self, x):
        # x: tensor of shape (batch_size, input_size)
        hidden = x

        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)

        # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = torch.flip(backward_prediction_hidden, dims=[-1])

        # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = torch.sigmoid(gate_hidden)

        # Combine forward and backward predictions
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden


        return out