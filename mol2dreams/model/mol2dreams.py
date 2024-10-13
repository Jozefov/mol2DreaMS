import torch
import torch.nn as nn
from mol2dreams.model.HeadLayer import IdentityHead


class Mol2DreaMS(nn.Module):
    def __init__(self, input_layer, global_input_layer=None, body_layer=None, head_layer=None):
        """
        Initializes the Mol2DreaMS model.

        Args:
            input_layer (nn.Module): The graph input layer.
            global_input_layer (nn.Module, optional): The global input layer. Defaults to None.
            body_layer (nn.Module): The body layer.
            head_layer (nn.Module): The head layer.
        """
        super(Mol2DreaMS, self).__init__()
        self.input_layer = input_layer
        self.global_input_layer = global_input_layer
        self.body_layer = body_layer
        self.head_layer = head_layer

    def forward(self, batch):
        """
        Forward pass for Mol2DreaMS.

        Args:
            batch (torch_geometric.data.Data): Batch of data.

        Returns:
            torch.Tensor: Output from the head layer.
        """
        # Process graph features
        graph_embeddings = self.input_layer(batch)  # Shape depends on InputLayer

        # Process global features if available
        if self.global_input_layer is not None and hasattr(batch, 'global_features'):

            global_features = batch.global_features.float()  # Shape: [batch_size, num_encoded_features]

            # Pass through GlobalInputLayer
            global_embeddings = self.global_input_layer(global_features)  # Shape: [batch_size, output_size]
        else:
            global_embeddings = None

        # Pass to BodyLayer
        processed_embeddings = self.body_layer(graph_embeddings, batch.batch, global_embeddings)  # Shape: [batch_size, embedding_size]

        # Pass to HeadLayer
        output = self.head_layer(processed_embeddings)  # Shape: [batch_size, output_size]

        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    