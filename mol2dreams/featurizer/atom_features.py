import torch
from rdkit import Chem
from mol2dreams.featurizer.featurize import one_hot_encode

class AtomFeaturizer:
    def __init__(self, config):
        """
        Initializes the AtomFeaturizer with specified configurations.

        Args:
            config (dict): Configuration dictionary specifying which features to extract.
        """
        self.config = config
        self.PERMITTED_ATOMS = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe',
                                'As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd',
                                'Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In',
                                'Mn','Zr','Cr','Pt','Hg','Pb', 'Unknown']
        self.atom_dict = {elem: idx for idx, elem in enumerate(self.PERMITTED_ATOMS)}
        self.num_atom_types = len(self.PERMITTED_ATOMS)

        self.hybridization_types = [
            Chem.HybridizationType.UNSPECIFIED,
            Chem.HybridizationType.S,
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
            Chem.HybridizationType.OTHER
        ]
        self.hybridization_dict = {htype: idx for idx, htype in enumerate(self.hybridization_types)}
        self.num_hybridization = len(self.hybridization_types)

    def featurize(self, atom):
        """
        Extracts features from an RDKit atom object.

        Args:
            atom (rdkit.Chem.rdchem.Atom): RDKit atom object.

        Returns:
            torch.Tensor: Feature tensor for the atom.
        """
        features = []

        # Atom type one-hot
        atom_symbol = atom.GetSymbol()
        atom_type = self.atom_dict.get(atom_symbol, self.atom_dict['Unknown'])
        atom_type_one_hot = one_hot_encode(atom_type, self.num_atom_types, allow_unknown=True)
        features.append(atom_type_one_hot)

        # Total valence one-hot
        total_valence = atom.GetTotalValence()
        total_valence_one_hot = one_hot_encode(total_valence, num_classes=8, allow_unknown=True)
        features.append(total_valence_one_hot)

        # Aromaticity
        if self.config.get('aromatic', True):
            aromatic = torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
            features.append(aromatic)

        # Hybridization one-hot
        if self.config.get('hybridization', True):
            hybridization = atom.GetHybridization()
            hybridization_idx = self.hybridization_dict.get(hybridization, self.hybridization_dict['OTHER'])
            hybridization_one_hot = one_hot_encode(hybridization_idx, self.num_hybridization, allow_unknown=False)
            features.append(hybridization_one_hot)

        # Formal charge one-hot (shifted to handle negative values)
        if self.config.get('formal_charge', True):
            formal_charge = atom.GetFormalCharge()
            formal_charge_shifted = formal_charge + 2  # Shift to make it non-negative
            formal_charge_one_hot = one_hot_encode(formal_charge_shifted, num_classes=5, allow_unknown=False)
            features.append(formal_charge_one_hot)

        # Default valence one-hot
        if self.config.get('default_valence', True):
            default_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
            default_valence_one_hot = one_hot_encode(default_valence, num_classes=8, allow_unknown=True)
            features.append(default_valence_one_hot)

        # Ring sizes
        if self.config.get('ring_size', False):
            ring_sizes = [atom.IsInRingSize(r) for r in range(3, 8)]
            ring_size_tensor = torch.tensor(ring_sizes, dtype=torch.float)
            features.append(ring_size_tensor)

        # Number of Hydrogens one-hot
        if self.config.get('hydrogen_count', False):
            attached_H = sum([1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1])
            explicit = atom.GetNumExplicitHs()
            implicit = atom.GetNumImplicitHs()
            H_num = attached_H + explicit + implicit
            H_num_one_hot = one_hot_encode(H_num, num_classes=6, allow_unknown=False)
            features.append(H_num_one_hot)

        # Concatenate all features
        feature_tensor = torch.cat(features, dim=0)
        return feature_tensor