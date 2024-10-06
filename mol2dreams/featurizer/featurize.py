import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from mol2dreams.featurizer.atom_features import AtomFeaturizer
from mol2dreams.featurizer.bond_features import BondFeaturizer

class MoleculeFeaturizer:
    def __init__(self, atom_config, bond_config, spectrum_embedding_size=1024):
        self.atom_featurizer = AtomFeaturizer(atom_config)
        self.bond_featurizer = BondFeaturizer(bond_config)
        self.spectrum_embedding_size = spectrum_embedding_size

    def featurize_molecule(self, mol, spectrum=None):
        if mol is None:
            return None

        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = self.atom_featurizer.featurize(atom)
            atom_features.append(atom_feat)
        X = torch.stack(atom_features, dim=0).float()  # Shape: [num_atoms, num_atom_features]

        # Extract bond features and edge indices
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # The graph is undirected, add both directions
            edge_index.extend([[i, j], [j, i]])

            bond_feat = self.bond_featurizer.featurize(bond)
            edge_features.extend([bond_feat, bond_feat])

        if len(edge_index) == 0:
            # Handle molecules with no bonds (single atoms)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((0, self.bond_featurizer.featurize(Chem.Bond()).shape[0]), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
            edge_features = torch.stack(edge_features, dim=0).float()  # Shape: [num_edges, num_bond_features]

        y = torch.zeros(self.spectrum_embedding_size, dtype=torch.float).unsqueeze(0)

        data = Data(x=X, edge_index=edge_index, edge_attr=edge_features, y=y)

        return data

    def featurize_dataset(self, nist_data):
        data_list = []
        for entry in nist_data:
            smiles = entry.get('smiles')
            spectrum = entry.get('spectrum', None)  

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            data = self.featurize_molecule(mol, spectrum)
            if data is not None:
                data_list.append(data)

        return data_list