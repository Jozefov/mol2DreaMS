from torch.utils.data import Dataset, DataLoader

import random

class MoleculeSpectrumDataset(Dataset):
    def __init__(self, molecule_data, spectral_embeddings, positive_indices, negative_indices, num_negatives=50, num_positives=1):
        """
        Args:
            molecule_data: List or array of molecular structures (e.g., SMILES strings).
            spectral_embeddings: Tensor of spectral embeddings [num_samples, embedding_dim].
            positive_indices: List of lists, where each sublist contains indices of positive spectra for each molecule.
            negative_indices: List of lists, where each sublist contains indices of negative spectra for each molecule.
            num_negatives: Number of negative samples per anchor.
            num_positives: Number of positive samples per anchor.
        """
        self.molecule_data = molecule_data
        self.spectral_embeddings = spectral_embeddings
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.num_negatives = num_negatives
        self.num_positives = num_positives

    def __len__(self):
        return len(self.molecule_data)

    def __getitem__(self, idx):
        molecule = self.molecule_data[idx]
        anchor_embedding = self.spectral_embeddings[idx]

        # Sample positives
        pos_indices = self.positive_indices[idx]
        if len(pos_indices) < self.num_positives:
            # If not enough positives, sample with replacement
            sampled_pos_indices = random.choices(pos_indices, k=self.num_positives)
        else:
            sampled_pos_indices = random.sample(pos_indices, self.num_positives)
        positive_embeddings = self.spectral_embeddings[sampled_pos_indices]

        # Sample negatives
        neg_indices = self.negative_indices[idx]
        if len(neg_indices) < self.num_negatives:
            sampled_neg_indices = random.choices(neg_indices, k=self.num_negatives)
        else:
            sampled_neg_indices = random.sample(neg_indices, self.num_negatives)
        negative_embeddings = self.spectral_embeddings[sampled_neg_indices]

        sample = {
            'anchor': anchor_embedding,
            'positives': positive_embeddings,
            'negatives': negative_embeddings
        }
        return sample