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
        self.epoch = 0
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
            total_cosine_sim_normalized = 0.0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Compute cosine similarity
                cosine_sim, cosine_sim_normalized = cosine_similarity_metric(outputs, batch.y)
                total_cosine_sim += cosine_sim
                total_cosine_sim_normalized += cosine_sim_normalized

            avg_loss = total_loss / len(self.train_loader)
            avg_cosine_sim = total_cosine_sim / len(self.train_loader)
            avg_cosine_sim_normalized = total_cosine_sim_normalized / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('CosineSimilarity/train', avg_cosine_sim, epoch)
            self.writer.add_scalar('CosineSimilarityNormalized/train', avg_cosine_sim_normalized, epoch)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Cosine Sim: {avg_cosine_sim:.4f},"
                  f"Cosine Sim Normalized: {avg_cosine_sim_normalized:.4f}")

            # Validation
            if self.val_loader and (epoch + 1) % self.validate_every == 0:
                val_loss, val_cosine_sim, val_cosine_sim_normalized = self.validate()
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('CosineSimilarity/val', val_cosine_sim, epoch)
                self.writer.add_scalar('CosineSimilarityNormalized/train', val_cosine_sim_normalized, epoch)
                print(f"Validation Loss: {val_loss:.4f}, Validation Cosine Sim: {val_cosine_sim:.4f},"
                      f"Validation Cosine Sim Normalized: {val_cosine_sim_normalized:.4f}")

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
        total_cosine_sim_normalized = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch.y)
                total_loss += loss.item()

                # Compute cosine similarity
                cosine_sim, cosine_sim_normalized = cosine_similarity_metric(outputs, batch.y)
                total_cosine_sim += cosine_sim
                total_cosine_sim_normalized += cosine_sim_normalized

        avg_loss = total_loss / len(self.val_loader)
        avg_cosine_sim = total_cosine_sim / len(self.val_loader)
        avg_cosine_sim_normalized = total_cosine_sim_normalized / len(self.val_loader)
        return avg_loss, avg_cosine_sim, avg_cosine_sim_normalized

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
        total_cosine_sim_normalized = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch.y)
                total_loss += loss.item()

                # Compute cosine similarity
                cosine_sim, cosine_sim_normalized = cosine_similarity_metric(outputs, batch.y)
                total_cosine_sim += cosine_sim
                total_cosine_sim_normalized += cosine_sim_normalized

        avg_loss = total_loss / len(self.test_loader)
        avg_cosine_sim = total_cosine_sim / len(self.test_loader)
        avg_cosine_sim_normalized = total_cosine_sim_normalized / len(self.test_loader)
        print(f"Test Loss: {avg_loss:.4f}, Test Cosine Sim: {avg_cosine_sim:.4f},"
              f"Test Cosine Sim Normalized: {avg_cosine_sim_normalized:.4f}")

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
            total_pos_similarity = 0.0
            total_neg_similarity = 0.0
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
                pos_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
                neg_similarities = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
                total_pos_similarity += pos_similarities.mean().item()
                total_neg_similarity += neg_similarities.mean().item()

            avg_loss = total_loss / len(self.train_loader)
            avg_pos_similarity = total_pos_similarity / len(self.train_loader)
            avg_neg_similarity = total_neg_similarity / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('CosineSimilarity/pos_train', avg_pos_similarity, epoch)
            self.writer.add_scalar('CosineSimilarity/neg_train', avg_neg_similarity, epoch)
            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Pos Sim: {avg_pos_similarity:.4f}, Neg Sim: {avg_neg_similarity:.4f}")

            # Validation
            if self.val_loader and (epoch + 1) % self.validate_every == 0:
                val_loss, val_pos_sim, val_neg_sim = self.validate()
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('CosineSimilarity/pos_val', val_pos_sim, epoch)
                self.writer.add_scalar('CosineSimilarity/neg_val', val_neg_sim, epoch)
                print(f"Validation Loss: {val_loss:.4f}, Pos Sim: {val_pos_sim:.4f}, Neg Sim: {val_neg_sim:.4f}")

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
        total_pos_similarity = 0.0
        total_neg_similarity = 0.0
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

                # Compute cosine similarity
                pos_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
                neg_similarities = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
                total_pos_similarity += pos_similarities.mean().item()
                total_neg_similarity += neg_similarities.mean().item()

        avg_loss = total_loss / len(self.val_loader)
        avg_pos_similarity = total_pos_similarity / len(self.val_loader)
        avg_neg_similarity = total_neg_similarity / len(self.val_loader)
        return avg_loss, avg_pos_similarity, avg_neg_similarity

    def test(self):
        if self.test_loader is None:
            print("Test loader is not provided.")
            return
        self.model.eval()
        total_loss = 0.0
        total_pos_similarity = 0.0
        total_neg_similarity = 0.0
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
                pos_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
                neg_similarities = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
                total_pos_similarity += pos_similarities.mean().item()
                total_neg_similarity += neg_similarities.mean().item()

        avg_loss = total_loss / len(self.val_loader)
        avg_pos_similarity = total_pos_similarity / len(self.val_loader)
        avg_neg_similarity = total_neg_similarity / len(self.val_loader)
        print(f"Test Loss: {avg_loss:.4f}, Pos Cosine: {avg_pos_similarity:.4f}, Neg Cosine: {avg_neg_similarity:.4f}")

class AdversarialTrainer(Trainer):
    def __init__(self, discriminator=None, discriminator_optimizer=None, lambda_triplet=1.0,
                 lambda_adv=1.0, lambda_gp=10.0, use_gradient_penalty=True, **kwargs):
        super(AdversarialTrainer, self).__init__(**kwargs)
        self.discriminator = discriminator.to(self.device) if discriminator else None
        self.discriminator_optimizer = discriminator_optimizer
        self.lambda_triplet = lambda_triplet
        self.lambda_adv = lambda_adv
        self.lambda_gp = lambda_gp
        self.use_gradient_penalty = use_gradient_penalty
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.model.train()
            if self.discriminator:
                self.discriminator.train()
            total_loss = 0.0
            total_triplet_loss = 0.0
            total_adv_loss = 0.0
            total_disc_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                anchor_batch, positive_batch, negative_batch = batch

                anchor_batch = anchor_batch.to(self.device)
                positive_batch = positive_batch.to(self.device)
                negative_batch = negative_batch.to(self.device)

                # Update Discriminator
                if self.discriminator:
                    self.discriminator_optimizer.zero_grad()

                    # Real embeddings (E_S_real): positive embeddings passed through the model
                    E_S_real = positive_batch.y.detach()
                    D_real = self.discriminator(E_S_real)

                    # Fake embeddings (E_M): embeddings from the molecular encoder (anchor)
                    E_M = self.model(anchor_batch)
                    D_fake = self.discriminator(E_M.detach())

                    real_labels = torch.ones_like(D_real)
                    fake_labels = torch.zeros_like(D_fake)

                    L_disc_real = self.bce_loss(D_real, real_labels)
                    L_disc_fake = self.bce_loss(D_fake, fake_labels)
                    L_disc = L_disc_real + L_disc_fake

                    if self.use_gradient_penalty:
                        gp = compute_gradient_penalty(
                            self.discriminator,
                            E_S_real.data,
                            E_M.data,
                            device=self.device
                        )
                        L_disc += self.lambda_gp * gp

                    # Update discriminator
                    L_disc.backward()
                    self.discriminator_optimizer.step()
                else:
                    L_disc = torch.tensor(0.0)

                # Update Molecular Encoder
                self.optimizer.zero_grad()

                # Compute E_M again for current parameters
                E_M = self.model(anchor_batch)

                # Compute D_fake for adversarial loss
                if self.discriminator:
                    D_fake_for_generator = self.discriminator(E_M)
                    real_labels = torch.ones_like(D_fake_for_generator)
                    L_adv = self.bce_loss(D_fake_for_generator, real_labels)
                else:
                    L_adv = torch.tensor(0.0).to(self.device)

                # Compute triplet loss
                pos_embeddings = self.model(positive_batch)
                neg_embeddings = self.model(negative_batch)
                L_triplet = self.loss_fn(E_M, pos_embeddings.detach(), neg_embeddings.detach())

                # Compute total loss
                total_loss_batch = self.lambda_triplet * L_triplet + self.lambda_adv * L_adv

                # Update molecular encoder
                total_loss_batch.backward()
                self.optimizer.step()

                total_loss += total_loss_batch.item()
                total_triplet_loss += L_triplet.item()
                total_adv_loss += L_adv.item()
                total_disc_loss += L_disc.item()

                # Optionally print progress
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Batch [{batch_idx}/{len(self.train_loader)}], '
                          f'Total Loss: {total_loss_batch.item():.4f}, '
                          f'Triplet Loss: {L_triplet.item():.4f}, '
                          f'Adversarial Loss: {L_adv.item():.4f}, '
                          f'Discriminator Loss: {L_disc.item():.4f}')

            # Compute average losses
            avg_total_loss = total_loss / len(self.train_loader)
            avg_triplet_loss = total_triplet_loss / len(self.train_loader)
            avg_adv_loss = total_adv_loss / len(self.train_loader)
            avg_disc_loss = total_disc_loss / len(self.train_loader)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train_total', avg_total_loss, epoch)
            self.writer.add_scalar('Loss/train_triplet', avg_triplet_loss, epoch)
            self.writer.add_scalar('Loss/train_adv', avg_adv_loss, epoch)
            self.writer.add_scalar('Loss/train_disc', avg_disc_loss, epoch)

            print(f'Epoch [{epoch+1}/{self.epochs}] Completed. '
                  f'Avg Total Loss: {avg_total_loss:.4f}, '
                  f'Avg Triplet Loss: {avg_triplet_loss:.4f}, '
                  f'Avg Adversarial Loss: {avg_adv_loss:.4f}, '
                  f'Avg Discriminator Loss: {avg_disc_loss:.4f}')

            # Validation
            if self.val_loader and (epoch + 1) % self.validate_every == 0:
                val_loss = self.validate()
                print(f'Validation Loss after Epoch {epoch+1}: {val_loss:.4f}')

                # Save best model
                if self.save_best_only and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(best=True)

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_adv_loss = 0.0
        total_pos_similarity = 0.0
        total_neg_similarity = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                anchor_batch, positive_batch, negative_batch = batch

                anchor_batch = anchor_batch.to(self.device)
                positive_batch = positive_batch.to(self.device)
                negative_batch = negative_batch.to(self.device)

                E_M = self.model(anchor_batch)

                # Compute discriminator output for E_M
                if self.discriminator:
                    D_fake = self.discriminator(E_M)
                    real_labels = torch.ones_like(D_fake)
                    L_adv = self.bce_loss(D_fake, real_labels)
                else:
                    L_adv = torch.tensor(0.0)

                # Compute triplet loss
                pos_embeddings = self.model(positive_batch).detach()
                neg_embeddings = self.model(negative_batch).detach()
                L_triplet = self.loss_fn(E_M, pos_embeddings, neg_embeddings)

                # Compute total loss
                total_loss_batch = self.lambda_triplet * L_triplet + self.lambda_adv * L_adv

                total_loss += total_loss_batch.item()
                total_triplet_loss += L_triplet.item()
                total_adv_loss += L_adv.item()

                pos_similarities = F.cosine_similarity(E_M, pos_embeddings, dim=1)
                neg_similarities = F.cosine_similarity(E_M, neg_embeddings, dim=1)
                total_pos_similarity += pos_similarities.mean().item()
                total_neg_similarity += neg_similarities.mean().item()

        avg_loss = total_loss / len(self.val_loader)
        avg_triplet_loss = total_triplet_loss / len(self.val_loader)
        avg_adv_loss = total_adv_loss / len(self.val_loader)
        avg_pos_similarity = total_pos_similarity / len(self.val_loader)
        avg_neg_similarity = total_neg_similarity / len(self.val_loader)

        self.writer.add_scalar('Loss/val_total', avg_loss, self.epoch)
        self.writer.add_scalar('Loss/val_triplet', avg_triplet_loss, self.epoch)
        self.writer.add_scalar('Loss/val_adv', avg_adv_loss, self.epoch)
        self.writer.add_scalar('CosineSimilarity/pos_val', avg_pos_similarity, self.epoch)
        self.writer.add_scalar('CosineSimilarity/neg_val', avg_neg_similarity, self.epoch)

        print(f'Validation Loss: {avg_loss:.4f}, Triplet Loss: {avg_triplet_loss:.4f}, '
              f'Adversarial Loss: {avg_adv_loss:.4f}, Pos Sim: {avg_pos_similarity:.4f}, '
              f'Neg Sim: {avg_neg_similarity:.4f}')
        return avg_loss

    def test(self):
        if self.test_loader is None:
            print("Test loader is not provided.")
            return
        self.model.eval()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_adv_loss = 0.0
        total_pos_similarity = 0.0
        total_neg_similarity = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue

                anchor_batch, positive_batch, negative_batch = batch

                anchor_batch = anchor_batch.to(self.device)
                positive_batch = positive_batch.to(self.device)
                negative_batch = negative_batch.to(self.device)

                E_M = self.model(anchor_batch)

                # Compute discriminator output for E_M
                if self.discriminator:
                    D_fake = self.discriminator(E_M)
                    real_labels = torch.ones_like(D_fake)
                    L_adv = self.bce_loss(D_fake, real_labels)
                else:
                    L_adv = torch.tensor(0.0)

                # Compute triplet loss
                pos_embeddings = self.model(positive_batch).detach()
                neg_embeddings = self.model(negative_batch).detach()
                L_triplet = self.loss_fn(E_M, pos_embeddings, neg_embeddings)

                # Compute total loss
                total_loss_batch = self.lambda_triplet * L_triplet + self.lambda_adv * L_adv

                total_loss += total_loss_batch.item()
                total_triplet_loss += L_triplet.item()
                total_adv_loss += L_adv.item()

                pos_similarities = F.cosine_similarity(E_M, pos_embeddings, dim=1)
                neg_similarities = F.cosine_similarity(E_M, neg_embeddings, dim=1)
                total_pos_similarity += pos_similarities.mean().item()
                total_neg_similarity += neg_similarities.mean().item()

        avg_loss = total_loss / len(self.test_loader)
        avg_triplet_loss = total_triplet_loss / len(self.test_loader)
        avg_adv_loss = total_adv_loss / len(self.test_loader)
        avg_pos_similarity = total_pos_similarity / len(self.test_loader)
        avg_neg_similarity = total_neg_similarity / len(self.test_loader)

        self.writer.add_scalar('Loss/test_total', avg_loss, self.epoch)
        self.writer.add_scalar('Loss/test_triplet', avg_triplet_loss, self.epoch)
        self.writer.add_scalar('Loss/test_adv', avg_adv_loss, self.epoch)
        self.writer.add_scalar('CosineSimilarity/pos_test', avg_pos_similarity, self.epoch)
        self.writer.add_scalar('CosineSimilarity/neg_test', avg_neg_similarity, self.epoch)

        print(f'Test Loss: {avg_loss:.4f}, Triplet Loss: {avg_triplet_loss:.4f}, '
              f'Adversarial Loss: {avg_adv_loss:.4f}, Pos Sim: {avg_pos_similarity:.4f}, '
              f'Neg Sim: {avg_neg_similarity:.4f}')

# class ContrastiveTrainer:
#     def __init__(self, config):
#         """
#         Initializes the Trainer.
#
#         Args:
#             config: A dictionary or argparse.Namespace containing training configurations.
#         """
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.config = config
#
#         # Initialize models
#         self.molecular_encoder = Mol2DreaMS(**config['encoder_params']).to(self.device)
#         self.discriminator = Discriminator(embedding_dim=config['embedding_dim']).to(self.device)
#
#         # Initialize losses
#         self.combined_loss_fn = CombinedLoss(
#             lambda_adv=config['lambda_adv'],
#             lambda_con=config['lambda_con'],
#             temperature=config.get('temperature', 0.07),
#             margin=config.get('margin', 1.0),
#             loss_type=config['loss_type']
#         ).to(self.device)
#
#         # Initialize optimizers
#         self.encoder_optimizer = optim.Adam(self.molecular_encoder.parameters(), lr=config['encoder_lr'])
#         self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['discriminator_lr'])
#
#         # Initialize datasets and dataloaders
#         self.train_loader, self.valid_loader, self.test_loader = self._create_dataloaders()
#
#         # Logger
#         # self.logger = setup_logger(config.get('log_file'))
#
#     def _create_dataloaders(self):
#         # Load your data here
#         # For example:
#         train_dataset = MoleculeSpectrumDataset(
#             molecule_data=self.config['train_molecule_data'],
#             spectral_embeddings=self.config['spectral_embeddings'],
#             positive_indices=self.config['train_positive_indices'],
#             negative_indices=self.config['train_negative_indices'],
#             num_negatives=self.config['num_negatives'],
#             num_positives=self.config['num_positives']
#         )
#
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.config['batch_size'],
#             shuffle=True,
#             num_workers=self.config.get('num_workers', 4),
#             collate_fn=MoleculeSpectrumDataset.collate_fn
#         )
#
#         # Similarly for valid_loader and test_loader
#         # You need to provide valid_molecule_data, valid_positive_indices, etc.
#
#         valid_loader = None  # Replace with actual DataLoader
#         test_loader = None   # Replace with actual DataLoader
#
#         return train_loader, valid_loader, test_loader
#
#     def train(self):
#         num_epochs = self.config['num_epochs']
#         lambda_gp = self.config.get('lambda_gp', 10.0)  # Gradient penalty coefficient
#
#         for epoch in range(num_epochs):
#             self.molecular_encoder.train()
#             self.discriminator.train()
#
#             for batch_idx, batch in enumerate(self.train_loader):
#                 # Move data to device
#                 molecules = batch['molecules']  # Adjust as needed
#                 positives = batch['positives'].to(self.device)   # [B, P, E]
#                 negatives = batch['negatives'].to(self.device)   # [B, N, E]
#
#                 # Process molecules through encoder
#                 E_M = self.molecular_encoder(molecules)  # [B, E]
#
#                 # Reshape positives and negatives if necessary
#                 E_S_positive = positives  # [B, P, E]
#                 E_S_negative = negatives  # [B, N, E]
#
#                 # --- Step 1: Update Discriminator ---
#                 self.discriminator_optimizer.zero_grad()
#
#                 # Real embeddings (from DreaMS)
#                 E_S_real = E_S_positive.view(-1, E_S_positive.size(-1))  # Flattened [B * P, E]
#                 D_real = self.discriminator(E_S_real)
#
#                 # Fake embeddings (detach to prevent gradients flowing back to encoder)
#                 D_fake = self.discriminator(E_M.detach())
#
#                 # Compute discriminator loss
#                 real_labels = torch.ones_like(D_real)
#                 fake_labels = torch.zeros_like(D_fake)
#
#                 bce_loss = nn.BCEWithLogitsLoss()
#
#                 L_disc_real = bce_loss(D_real, real_labels)
#                 L_disc_fake = bce_loss(D_fake, fake_labels)
#                 L_disc = L_disc_real + L_disc_fake
#
#                 # Gradient penalty
#                 gp = compute_gradient_penalty(
#                     self.discriminator,
#                     E_S_real.data,
#                     E_M.data,
#                     device=self.device
#                 )
#                 L_disc += lambda_gp * gp
#
#                 # Backpropagation and update discriminator
#                 L_disc.backward()
#                 self.discriminator_optimizer.step()
#
#                 # --- Step 2: Update Molecular Encoder ---
#                 self.encoder_optimizer.zero_grad()
#
#                 # Compute D_fake again (without detaching) for adversarial loss
#                 D_fake_for_generator = self.discriminator(E_M)
#
#                 # Compute combined loss
#                 total_loss, L_adv, L_con, _ = self.combined_loss_fn(
#                     E_M,
#                     E_S_positive,
#                     E_S_negative,
#                     D_real.detach(),           # Detach to prevent gradients through discriminator
#                     D_fake_for_generator
#                 )
#
#                 # Backpropagation and update encoder
#                 total_loss.backward()
#                 self.encoder_optimizer.step()
#
#                 # Logging
#                 if batch_idx % self.config.get('log_interval', 10) == 0:
#                     self.logger.info(
#                         f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(self.train_loader)}] "
#                         f"Total Loss: {total_loss.item():.4f}, L_adv: {L_adv.item():.4f}, "
#                         f"L_con: {L_con.item():.4f}, L_disc: {L_disc.item():.4f}, GP: {gp.item():.4f}"
#                     )
#
#             # Optionally validate and save model checkpoints here
#
#     def validate(self):
#         # Implement validation logic here
#         pass
#
#     def test(self):
#         # Implement testing logic here
#         pass