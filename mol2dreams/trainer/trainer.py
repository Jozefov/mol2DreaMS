import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mol2dreams.model import Mol2DreaMS
from mol2dreams.model.discriminator import Discriminator
from mol2dreams.model.loss import CombinedLoss
from mol2dreams.datasets.mocule_spectrum_dataset import MoleculeSpectrumDataset
from mol2dreams.utils.gradient_penalty import compute_gradient_penalty
from mol2dreams.utils.monitoring import setup_logger

class Trainer:
    def __init__(self, config):
        """
        Initializes the Trainer.

        Args:
            config: A dictionary or argparse.Namespace containing training configurations.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Initialize models
        self.molecular_encoder = Mol2DreaMS(**config['encoder_params']).to(self.device)
        self.discriminator = Discriminator(embedding_dim=config['embedding_dim']).to(self.device)

        # Initialize losses
        self.combined_loss_fn = CombinedLoss(
            lambda_adv=config['lambda_adv'],
            lambda_con=config['lambda_con'],
            temperature=config.get('temperature', 0.07),
            margin=config.get('margin', 1.0),
            loss_type=config['loss_type']
        ).to(self.device)

        # Initialize optimizers
        self.encoder_optimizer = optim.Adam(self.molecular_encoder.parameters(), lr=config['encoder_lr'])
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['discriminator_lr'])

        # Initialize datasets and dataloaders
        self.train_loader, self.valid_loader, self.test_loader = self._create_dataloaders()

        # Logger
        self.logger = setup_logger(config.get('log_file'))

    def _create_dataloaders(self):
        # Load your data here
        # For example:
        train_dataset = MoleculeSpectrumDataset(
            molecule_data=self.config['train_molecule_data'],
            spectral_embeddings=self.config['spectral_embeddings'],
            positive_indices=self.config['train_positive_indices'],
            negative_indices=self.config['train_negative_indices'],
            num_negatives=self.config['num_negatives'],
            num_positives=self.config['num_positives']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=MoleculeSpectrumDataset.collate_fn
        )

        # Similarly for valid_loader and test_loader
        # You need to provide valid_molecule_data, valid_positive_indices, etc.

        valid_loader = None  # Replace with actual DataLoader
        test_loader = None   # Replace with actual DataLoader

        return train_loader, valid_loader, test_loader

    def train(self):
        num_epochs = self.config['num_epochs']
        lambda_gp = self.config.get('lambda_gp', 10.0)  # Gradient penalty coefficient

        for epoch in range(num_epochs):
            self.molecular_encoder.train()
            self.discriminator.train()

            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                molecules = batch['molecules']  # Adjust as needed
                positives = batch['positives'].to(self.device)   # [B, P, E]
                negatives = batch['negatives'].to(self.device)   # [B, N, E]

                # Process molecules through encoder
                E_M = self.molecular_encoder(molecules)  # [B, E]

                # Reshape positives and negatives if necessary
                E_S_positive = positives  # [B, P, E]
                E_S_negative = negatives  # [B, N, E]

                # --- Step 1: Update Discriminator ---
                self.discriminator_optimizer.zero_grad()

                # Real embeddings (from DreaMS)
                E_S_real = E_S_positive.view(-1, E_S_positive.size(-1))  # Flattened [B * P, E]
                D_real = self.discriminator(E_S_real)

                # Fake embeddings (detach to prevent gradients flowing back to encoder)
                D_fake = self.discriminator(E_M.detach())

                # Compute discriminator loss
                real_labels = torch.ones_like(D_real)
                fake_labels = torch.zeros_like(D_fake)

                bce_loss = nn.BCEWithLogitsLoss()

                L_disc_real = bce_loss(D_real, real_labels)
                L_disc_fake = bce_loss(D_fake, fake_labels)
                L_disc = L_disc_real + L_disc_fake

                # Gradient penalty
                gp = compute_gradient_penalty(
                    self.discriminator,
                    E_S_real.data,
                    E_M.data,
                    device=self.device
                )
                L_disc += lambda_gp * gp

                # Backpropagation and update discriminator
                L_disc.backward()
                self.discriminator_optimizer.step()

                # --- Step 2: Update Molecular Encoder ---
                self.encoder_optimizer.zero_grad()

                # Compute D_fake again (without detaching) for adversarial loss
                D_fake_for_generator = self.discriminator(E_M)

                # Compute combined loss
                total_loss, L_adv, L_con, _ = self.combined_loss_fn(
                    E_M,
                    E_S_positive,
                    E_S_negative,
                    D_real.detach(),           # Detach to prevent gradients through discriminator
                    D_fake_for_generator
                )

                # Backpropagation and update encoder
                total_loss.backward()
                self.encoder_optimizer.step()

                # Logging
                if batch_idx % self.config.get('log_interval', 10) == 0:
                    self.logger.info(
                        f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(self.train_loader)}] "
                        f"Total Loss: {total_loss.item():.4f}, L_adv: {L_adv.item():.4f}, "
                        f"L_con: {L_con.item():.4f}, L_disc: {L_disc.item():.4f}, GP: {gp.item():.4f}"
                    )

            # Optionally validate and save model checkpoints here

    def validate(self):
        # Implement validation logic here
        pass

    def test(self):
        # Implement testing logic here
        pass