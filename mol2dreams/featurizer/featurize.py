import torch
import tqdm
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from mol2dreams.featurizer.atom_features import AtomFeaturizer
from mol2dreams.featurizer.bond_features import BondFeaturizer
from mol2dreams.featurizer.global_features import GlobalFeaturizer

class MoleculeFeaturizer:
    def __init__(self, atom_config, bond_config, global_config=None, spectrum_embedding_size=1024):
        """
        Initializes the MoleculeFeaturizer with specified configurations.

        Args:
            atom_config (dict): Configuration for atom features.
            bond_config (dict): Configuration for bond features.
            spectrum_embedding_size (int): Size of the embedding vector.
            extra_features (list of str): List of additional attributes to include.
        """
        self.atom_featurizer = AtomFeaturizer(atom_config)
        self.bond_featurizer = BondFeaturizer(bond_config)
        self.spectrum_embedding_size = spectrum_embedding_size

        if global_config is not None:
            self.global_featurizer = GlobalFeaturizer(global_config)
        else:
            self.global_featurizer = None

    def featurize_molecule(self, mol, embedding=None, extra_attrs=None):
        """
        Featurizes a single molecule into a PyTorch Geometric Data object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
            embedding (np.ndarray or torch.Tensor, optional): Embedding vector. Defaults to None.
            extra_attrs (dict, optional): Dictionary of extra attributes to include. Defaults to None.

        Returns:
            torch_geometric.data.Data: Featurized graph data object with embedding and extra attributes.
        """
        if mol is None:
            return None

        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = self.atom_featurizer.featurize(atom)
            atom_features.append(atom_feat)
        X = torch.stack(atom_features, dim=0).float()

        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Since the graph is undirected, add both directions
            edge_index.extend([[i, j], [j, i]])

            bond_feat = self.bond_featurizer.featurize(bond)
            edge_features.extend([bond_feat, bond_feat])

        if len(edge_index) == 0:
            # Handle molecules with no bonds ( single atoms)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            bond_feature_size = self.bond_featurizer.featurize(Chem.Bond()).shape[0]
            edge_features = torch.empty((0, bond_feature_size), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_features = torch.stack(edge_features, dim=0).float()  #

        if embedding is not None:
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if embedding.shape[-1] != self.spectrum_embedding_size:
                raise ValueError(
                    f"Embedding size mismatch: expected {self.spectrum_embedding_size}, got {embedding.shape[0]}")
            y = embedding
        else:
            y = torch.zeros(self.spectrum_embedding_size, dtype=torch.float)

        data = Data(x=X, edge_index=edge_index, edge_attr=edge_features, y=y)

        # Add global features as separate attributes if global_featurizer is available
        if self.global_featurizer and extra_attrs:
            try:
                global_features_tensor = self.global_featurizer.featurize(extra_attrs)
                data["global_features"] = global_features_tensor  # Each key corresponds to a global feature
            except Exception as e:
                print(f"Error featurizing global features for molecule: {e}")

        return data

    def featurize_dataset(self, nist_data, include_extra_attr=True):
        """
        Featurizes an entire dataset of molecules, assigning embeddings and extra attributes.
        IF embedding is not presented, it will be filled with tensor of zeroes

        Args:
            nist_data (list of dict): Each dict should contain 'smiles', 'embedding', and extra attributes.
            include_extra_attr (bool, optional): Whether to include additional attributes.
        Returns:
            list of torch_geometric.data.Data or None: List of featurized graph data objects.
        """
        data_list = []
        total = len(nist_data)
        for entry in tqdm.tqdm(nist_data, desc="Featurizing dataset", total=total):
            smiles = entry.get('smiles')
            embedding = entry.get('embedding')
            # Extract extra attributes
            extra_attrs = {k: v for k, v in entry.items() if k not in ['smiles', 'embedding']}

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Skipping invalid SMILES: {smiles}")
                data_list.append(None)
                continue

            try:
                if include_extra_attr:
                    data = self.featurize_molecule(mol, embedding=embedding, extra_attrs=extra_attrs)
                else:
                    data = self.featurize_molecule(mol, embedding=embedding, extra_attrs=None)
            except Exception as e:
                print(f"Error featurizing molecule with SMILES {smiles}: {e}")
                data = None

            data_list.append(data)

        return data_list