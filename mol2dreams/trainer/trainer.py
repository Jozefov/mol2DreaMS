import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from mol2dreams.model.mol2dreams import Mol2DreaMS
from mol2dreams.model.discriminator import Discriminator
from mol2dreams.trainer.loss import CombinedLoss
from mol2dreams.datasets.mocule_spectrum_dataset import MoleculeSpectrumDataset
from mol2dreams.utils.gradient_penalty import compute_gradient_penalty
from mol2dreams.utils.metrics import cosine_similarity_metric


class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader=None, test_loader=None,
                 device='cpu', log_dir='./logs', epochs=30, validate_every=1, save_every=1, save_best_only=True):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log_dir = log_dir
        self.epochs = epochs
        self.writer = SummaryWriter(log_dir=log_dir)
        self.validate_every = validate_every
        self.save_every = save_every
        self.save_best_only = save_best_only
        self.best_val_loss = np.inf

        # Create directory for saving models
        os.makedirs(self.log_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            total_cosine_sim = 0.0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Compute cosine similarity
                cosine_sim = cosine_similarity_metric(outputs, batch.y)
                total_cosine_sim += cosine_sim

            avg_loss = total_loss / len(self.train_loader)
            avg_cosine_sim = total_cosine_sim / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('CosineSimilarity/train', avg_cosine_sim, epoch)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Cosine Sim: {avg_cosine_sim:.4f}")

            # Validation
            if self.val_loader and (epoch + 1) % self.validate_every == 0:
                val_loss, val_cosine_sim = self.validate()
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('CosineSimilarity/val', val_cosine_sim, epoch)
                print(f"Validation Loss: {val_loss:.4f}, Validation Cosine Sim: {val_cosine_sim:.4f}")

                # Check if this is the best validation loss so far
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(best=True)

            # Save model checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_cosine_sim = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch.y)
                total_loss += loss.item()

                # Compute cosine similarity
                cosine_sim = cosine_similarity_metric(outputs, batch.y)
                total_cosine_sim += cosine_sim

        avg_loss = total_loss / len(self.val_loader)
        avg_cosine_sim = total_cosine_sim / len(self.val_loader)
        return avg_loss, avg_cosine_sim

    def save_checkpoint(self, best=False):
        if best and self.save_best_only:
            # Remove previous best model if exists
            best_model_path = os.path.join(self.log_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            # Save new best model
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {self.epoch+1} with validation loss {self.best_val_loss:.4f}")
        else:
            checkpoint_path = os.path.join(self.log_dir, f'model_epoch_{self.epoch+1}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    def test(self):
        if self.test_loader is None:
            print("Test loader is not provided.")
            return
        self.model.eval()
        total_loss = 0.0
        total_cosine_sim = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch.y)
                total_loss += loss.item()

                # Compute cosine similarity
                cosine_sim = cosine_similarity_metric(outputs, batch.y)
                total_cosine_sim += cosine_sim

        avg_loss = total_loss / len(self.test_loader)
        avg_cosine_sim = total_cosine_sim / len(self.test_loader)
        print(f"Test Loss: {avg_loss:.4f}, Test Cosine Sim: {avg_cosine_sim:.4f}")

class TripletTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader=None, test_loader=None,
                 device='cpu', log_dir='./logs', epochs=30, validate_every=1, save_every=1, save_best_only=True):
        super(TripletTrainer, self).__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            log_dir=log_dir,
            epochs=epochs,
            validate_every=validate_every,
            save_every=save_every,
            save_best_only=save_best_only
        )

    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            total_pos_distance = 0.0
            total_neg_distance = 0.0
            for batch in self.train_loader:
                if batch is None:
                    continue  # Skip if batch is empty due to filtering

                anchor_batch, positive_batch, negative_batch = batch

                # Move batches to device
                anchor_batch = anchor_batch.to(self.device)
                positive_batch = positive_batch.to(self.device)
                negative_batch = negative_batch.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                anchor_embeddings = self.model(anchor_batch)
                positive_embeddings = self.model(positive_batch)
                negative_embeddings = self.model(negative_batch)

                # Compute the triplet loss
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Compute distances
                pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
                neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings)
                total_pos_distance += pos_distances.mean().item()
                total_neg_distance += neg_distances.mean().item()

            avg_loss = total_loss / len(self.train_loader)
            avg_pos_distance = total_pos_distance / len(self.train_loader)
            avg_neg_distance = total_neg_distance / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Distance/pos_train', avg_pos_distance, epoch)
            self.writer.add_scalar('Distance/neg_train', avg_neg_distance, epoch)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Pos Dist: {avg_pos_distance:.4f}, Neg Dist: {avg_neg_distance:.4f}")

            # Validation
            if self.val_loader and (epoch + 1) % self.validate_every == 0:
                val_loss, val_pos_dist, val_neg_dist = self.validate()
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Distance/pos_val', val_pos_dist, epoch)
                self.writer.add_scalar('Distance/neg_val', val_neg_dist, epoch)
                print(f"Validation Loss: {val_loss:.4f}, Pos Dist: {val_pos_dist:.4f}, Neg Dist: {val_neg_dist:.4f}")

                # Check if this is the best validation loss so far
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(best=True)

            # Save model checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_pos_distance = 0.0
        total_neg_distance = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue  # Skip if batch is empty due to filtering

                anchor_batch, positive_batch, negative_batch = batch

                # Move batches to device
                anchor_batch = anchor_batch.to(self.device)
                positive_batch = positive_batch.to(self.device)
                negative_batch = negative_batch.to(self.device)

                # Forward pass
                anchor_embeddings = self.model(anchor_batch)
                positive_embeddings = self.model(positive_batch)
                negative_embeddings = self.model(negative_batch)

                # Compute the triplet loss
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

                total_loss += loss.item()

                # Compute distances
                pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
                neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings)
                total_pos_distance += pos_distances.mean().item()
                total_neg_distance += neg_distances.mean().item()

        avg_loss = total_loss / len(self.val_loader)
        avg_pos_distance = total_pos_distance / len(self.val_loader)
        avg_neg_distance = total_neg_distance / len(self.val_loader)
        return avg_loss, avg_pos_distance, avg_neg_distance

    def test(self):
        if self.test_loader is None:
            print("Test loader is not provided.")
            return
        self.model.eval()
        total_loss = 0.0
        total_pos_distance = 0.0
        total_neg_distance = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue  # Skip if batch is empty due to filtering

                anchor_batch, positive_batch, negative_batch = batch

                # Move batches to device
                anchor_batch = anchor_batch.to(self.device)
                positive_batch = positive_batch.to(self.device)
                negative_batch = negative_batch.to(self.device)

                # Forward pass
                anchor_embeddings = self.model(anchor_batch)
                positive_embeddings = self.model(positive_batch)
                negative_embeddings = self.model(negative_batch)

                # Compute the triplet loss
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

                total_loss += loss.item()

                # Compute distances
                pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
                neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings)
                total_pos_distance += pos_distances.mean().item()
                total_neg_distance += neg_distances.mean().item()

        avg_loss = total_loss / len(self.test_loader)
        avg_pos_distance = total_pos_distance / len(self.test_loader)
        avg_neg_distance = total_neg_distance / len(self.test_loader)
        print(f"Test Loss: {avg_loss:.4f}, Pos Dist: {avg_pos_distance:.4f}, Neg Dist: {avg_neg_distance:.4f}")

class ContrastiveTrainer:
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
        # self.logger = setup_logger(config.get('log_file'))

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