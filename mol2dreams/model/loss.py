import torch
import torch.nn as nn
import torch.nn.functional as F

# class InfoNCELoss(torch.nn.Module):
#     def __init__(self, temperature=0.07):
#         super(InfoNCELoss, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, anchor, positive, negatives):
#         """
#         anchor: Tensor of shape [batch_size, embedding_dim]
#         positive: Tensor of shape [batch_size, embedding_dim]
#         negatives: Tensor of shape [batch_size, num_negatives, embedding_dim]
#         """
#         # Normalize embeddings
#         anchor = F.normalize(anchor, dim=1)
#         positive = F.normalize(positive, dim=1)
#         negatives = F.normalize(negatives, dim=2)
#
#         # Compute positive similarities
#         pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # Shape: [batch_size]
#
#         # Compute negative similarities
#         neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature  # Shape: [batch_size, num_negatives]
#
#         # Concatenate positive and negative similarities
#         logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # Shape: [batch_size, 1 + num_negatives]
#
#         # Labels: positive is at index 0
#         labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)
#
#         # Compute cross-entropy loss
#         loss = F.cross_entropy(logits, labels)
#         return loss

class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        super(TripletMarginLoss, self).__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet margin loss.

        Args:
            anchor: Tensor of shape [batch_size, embedding_dim]
            positive: Tensor of shape [batch_size, embedding_dim]
            negative: Tensor of shape [batch_size, embedding_dim]

        Returns:
            loss: Scalar tensor containing the triplet margin loss.
        """
        loss = self.loss_fn(anchor, positive, negative)
        return loss

class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

    def forward(self, input1, input2, target):
        """
        Compute the cosine embedding loss.

        Args:
            input1: Tensor of shape [batch_size, embedding_dim]
            input2: Tensor of shape [batch_size, embedding_dim]
            target: Tensor of shape [batch_size], with 1 for similar pairs and -1 for dissimilar pairs

        Returns:
            loss: Scalar tensor containing the cosine embedding loss.
        """
        loss = self.loss_fn(input1, input2, target)
        return loss