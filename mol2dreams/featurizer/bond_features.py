import torch
from rdkit import Chem
from mol2dreams.featurizer.featurize import one_hot_encode

class BondFeaturizer:
    def __init__(self, config):
        """
        Initializes the BondFeaturizer with specified configurations.

        Args:
            config (dict): Configuration dictionary specifying which features to extract.
        """
        self.config = config
        self.BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.bond_dict = {bond_type: idx for idx, bond_type in enumerate(self.BOND_TYPES)}
        self.num_bond_types = len(self.BOND_TYPES)

        self.stereo_types = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        self.stereo_dict = {stype: idx for idx, stype in enumerate(self.stereo_types)}
        self.num_stereo_types = len(self.stereo_types)

    def featurize(self, bond):
        """
        Extracts features from an RDKit bond object.

        Args:
            bond (rdkit.Chem.rdchem.Bond): RDKit bond object.

        Returns:
            torch.Tensor: Feature tensor for the bond.
        """
        features = []

        # Bond type one-hot
        bond_type = bond.GetBondType()
        bond_type_idx = self.bond_dict.get(bond_type, self.num_bond_types)  # Assign to 'Unknown' if not found
        bond_type_one_hot = one_hot_encode(bond_type_idx, self.num_bond_types + 1, allow_unknown=False)
        features.append(bond_type_one_hot)

        # Conjugation
        if self.config.get('conjugated', True):
            conjugated = torch.tensor([bond.GetIsConjugated()], dtype=torch.float)
            features.append(conjugated)

        # In-ring
        if self.config.get('in_ring', True):
            in_ring = torch.tensor([bond.IsInRing()], dtype=torch.float)
            features.append(in_ring)

        # Stereochemistry one-hot
        if self.config.get('stereochemistry', False):
            stereo = str(bond.GetStereo())
            stereo_idx = self.stereo_dict.get(stereo, self.num_stereo_types - 1)  # Assign to 'STEREONONE' if not found
            stereo_one_hot = one_hot_encode(stereo_idx, self.num_stereo_types, allow_unknown=False)
            features.append(stereo_one_hot)

        # Concatenate all features
        feature_tensor = torch.cat(features, dim=0)
        return feature_tensor