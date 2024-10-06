from torch.utils.data import Dataset, DataLoader

class MoleculeSpectrumDataset(Dataset):
    def __init__(self, molecule_data, spectral_embeddings, positive_indices, negative_indices):
        """
        Args:
            molecule_data: List or array of molecular structures (e.g., SMILES strings).
            spectral_embeddings: Precomputed spectral embeddings corresponding to the molecules.
            positive_indices: List of lists, where each sublist contains indices of positive spectra for each molecule.
            negative_indices: List of lists, where each sublist contains indices of negative spectra for each molecule.
        """
        self.molecule_data = molecule_data
        self.spectral_embeddings = spectral_embeddings
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices

    def __len__(self):
        return len(self.molecule_data)

    def __getitem__(self, idx):
        molecule = self.molecule_data[idx]
        anchor_molecule = molecule  # This can be processed further as needed

        # Positive spectral embeddings
        pos_indices = self.positive_indices[idx]
        positive_embeddings = self.spectral_embeddings[pos_indices]

        # Negative spectral embeddings
        neg_indices = self.negative_indices[idx]
        negative_embeddings = self.spectral_embeddings[neg_indices]

        sample = {
            'molecule': anchor_molecule,
            'positive_embeddings': positive_embeddings,
            'negative_embeddings': negative_embeddings
        }
        return sample