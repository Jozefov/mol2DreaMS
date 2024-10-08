import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELossWrapper(nn.Module):
    def __init__(self):
        super(MSELossWrapper, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, negatives):
        """
        anchor: Tensor of shape [batch_size, embedding_dim]
        positives: Tensor of shape [batch_size, num_positives, embedding_dim]
        negatives: Tensor of shape [batch_size, num_negatives, embedding_dim]
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)                             # [B, E]
        positives = F.normalize(positives, dim=2)                       # [B, P, E]
        negatives = F.normalize(negatives, dim=2)                       # [B, N, E]

        batch_size, num_positives, embedding_dim = positives.shape
        num_negatives = negatives.shape[1]

        # Compute positive similarities
        # Average over multiple positives, but I will for now work with just 1
        pos_sim = torch.bmm(positives, anchor.unsqueeze(2)).squeeze(2)  # [B, 1]
        pos_sim = pos_sim.mean(dim=1) / self.temperature               # [B]

        # Compute negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # [B, N]
        neg_sim = neg_sim / self.temperature

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)      # [B, 1 + N]

        # Labels: positive is at index 0
        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

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

class CombinedLoss(nn.Module):
    def __init__(self, lambda_adv=0.5, lambda_con=0.5, temperature=0.07, margin=1.0, loss_type='InfoNCE', use_logits=True):
        """
        Initializes the CombinedLoss module.

        Args:
            lambda_adv (float): Weight for adversarial loss.
            lambda_con (float): Weight for contrastive loss.
            temperature (float): Temperature parameter for InfoNCE.
            margin (float): Margin parameter for Triplet and Cosine Embedding Losses.
            loss_type (str): Type of contrastive loss to use ('InfoNCE', 'Triplet', 'CosineEmbedding').
            use_logits (bool): If True, assumes discriminator outputs are logits and uses BCEWithLogitsLoss.
                               If False, assumes outputs are probabilities and uses BCELoss.
        """
        super(CombinedLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_con = lambda_con
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type

        # Choose appropriate BCE loss based on discriminator output
        if use_logits:
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCELoss()

        if loss_type == 'InfoNCE':
            self.contrastive_loss_fn = InfoNCELoss(temperature=temperature)
        elif loss_type == 'Triplet':
            self.contrastive_loss_fn = TripletMarginLoss(margin=margin)
        elif loss_type == 'CosineEmbedding':
            self.contrastive_loss_fn = CosineEmbeddingLoss(margin=margin)
        else:
            raise ValueError('Invalid loss_type. Choose from "InfoNCE", "Triplet", or "CosineEmbedding".')

    def forward(self, E_M, E_S_positive, E_S_negative, D_real, D_fake):
        """
        Compute the combined loss.

        Args:
            E_M: Molecular embeddings from the encoder (batch_size, embedding_dim)
            E_S_positive: Positive spectral embeddings (batch_size, embedding_dim)
            E_S_negative: Negative spectral embeddings (batch_size, embedding_dim) or list/tensor of negatives
            D_real: Discriminator outputs on real spectral embeddings (batch_size)
            D_fake: Discriminator outputs on fake (molecular) embeddings (batch_size)

        Returns:
            total_loss: The total combined loss
            L_adv: Adversarial loss component
            L_con: Contrastive loss component
            L_disc: Discriminator loss
        """
        # Adversarial Losses
        # Discriminator Loss
        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)

        L_disc_real = self.bce_loss(D_real, real_labels)

        # Detach needed, to avoid flow of gradient to molecular encoder
        L_disc_fake = self.bce_loss(D_fake.detach(), fake_labels)
        L_disc = L_disc_real + L_disc_fake

        # Generator Adversarial Loss, here we want to flow gradient to molecular encoder
        L_adv = self.bce_loss(D_fake, real_labels)

        # Contrastive Loss
        if self.loss_type == 'InfoNCE':
            L_con = self.contrastive_loss_fn(E_M, E_S_positive, E_S_negative)
        elif self.loss_type == 'Triplet':
            L_con = self.contrastive_loss_fn(E_M, E_S_positive, E_S_negative)
        elif self.loss_type == 'CosineEmbedding':
            # For CosineEmbeddingLoss, targets are 1 for positives, -1 for negatives
            inputs1 = torch.cat([E_M, E_M], dim=0)
            inputs2 = torch.cat([E_S_positive, E_S_negative], dim=0)
            targets = torch.cat([torch.ones(E_M.size(0)), -torch.ones(E_M.size(0))]).to(E_M.device)
            L_con = self.contrastive_loss_fn(inputs1, inputs2, targets)
        else:
            raise ValueError('Invalid loss_type.')

        # Total Loss
        total_loss = self.lambda_adv * L_adv + self.lambda_con * L_con

        return total_loss, L_adv, L_con, L_disc