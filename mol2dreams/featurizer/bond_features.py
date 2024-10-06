# bond_featurizer.py

import torch
from rdkit import Chem
from mol2dreams.utils.data import one_hot_encode

class BondFeaturizer:
    def __init__(self, config):
        """
        Initializes the BondFeaturizer with specified configurations.

        Args:
            config (dict): Configuration dictionary specifying which features to extract.
        """
        self.config = config['features']

        # Define known bond types
        self.BOND_TYPES = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        self.bond_dict = {bond_type: idx for idx, bond_type in enumerate(self.BOND_TYPES)}
        self.num_bond_types = len(self.BOND_TYPES)

        # Define stereo types
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

        # 1. Bond Type One-Hot
        if self.config.get('bond_type', False):
            bond_type = bond.GetBondType()
            bond_type_idx = self.bond_dict.get(bond_type, self.num_bond_types)  # Assign to 'Unknown' if not found
            bond_type_one_hot = one_hot_encode(bond_type_idx, self.num_bond_types + 1)
            features.append(bond_type_one_hot)

        # 2. Conjugation
        if self.config.get('conjugated', False):
            conjugated = torch.tensor([bond.GetIsConjugated()], dtype=torch.float)
            features.append(conjugated)

        # 3. In-Ring
        if self.config.get('in_ring', False):
            in_ring = torch.tensor([bond.IsInRing()], dtype=torch.float)
            features.append(in_ring)

        # 4. Stereochemistry One-Hot
        if self.config.get('stereochemistry', False):
            stereo = str(bond.GetStereo())
            stereo_idx = self.stereo_dict.get(stereo, self.num_stereo_types - 1)  # Assign to 'STEREONONE' if not found
            stereo_one_hot = one_hot_encode(stereo_idx, self.num_stereo_types)
            features.append(stereo_one_hot)

        # Concatenate all features
        try:
            feature_tensor = torch.cat(features, dim=0)
        except RuntimeError as e:
            feature_shapes = [f.shape for f in features]
            raise RuntimeError(
                f"Error concatenating features for bond between atoms {bond.GetBeginAtomIdx()} and {bond.GetEndAtomIdx()}: {e}\n"
                f"Feature shapes: {feature_shapes}"
            )

        return feature_tensor