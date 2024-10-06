import torch
from rdkit import Chem
from mol2dreams.utils.data import one_hot_encode

class AtomFeaturizer:
    def __init__(self, config):
        """
        Initializes the AtomFeaturizer with specified configurations.

        Args:
            config (dict): Configuration dictionary specifying which features to extract.
        """
        self.config = config['features']
        self.feature_attributes = config.get('feature_attributes', {})

        # Define the full list of permitted atoms, including 'Unknown' at the end
        self.PERMITTED_ATOMS = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
            'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
            'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
            'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]

        # Extract atom_symbol attributes
        atom_symbol_attrs = self.feature_attributes.get('atom_symbol', {})
        self.top_n_atoms = atom_symbol_attrs.get('top_n_atoms', 12)
        self.include_other = atom_symbol_attrs.get('include_other', True)

        # Validate top_n_atoms
        if self.top_n_atoms > len(self.PERMITTED_ATOMS) - 1:
            raise ValueError(f"'top_n_atoms' cannot exceed {len(self.PERMITTED_ATOMS) - 1}")

        # Create a subset of PERMITTED_ATOMS based on top_n_atoms
        self.recognized_atoms = self.PERMITTED_ATOMS[:self.top_n_atoms]
        # Always include 'Unknown' class
        self.recognized_atoms.append('Unknown')  # Index = top_n_atoms

        # Create atom_dict mapping atom symbols to indices
        self.atom_dict = {elem: idx for idx, elem in enumerate(self.recognized_atoms)}
        self.num_atom_types = len(self.atom_dict)  # top_n_atoms + 1

        # Define hybridization types
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

        # 1. Atom Symbol One-Hot
        if self.config.get('atom_symbol', False):
            atom_symbol = atom.GetSymbol()
            # Map to 'Unknown' if not in top_n_atoms
            atom_type = self.atom_dict.get(atom_symbol, self.atom_dict['Unknown'])
            atom_symbol_one_hot = one_hot_encode(atom_type, self.num_atom_types)
            features.append(atom_symbol_one_hot)

        # 2. Total Valence One-Hot
        if self.config.get('total_valence', False):
            total_valence = atom.GetTotalValence()
            # Assuming valence ranges from 0 to 7 (num_classes=8)
            total_valence = min(max(total_valence, 0), 7)  # Clamp to [0,7]
            total_valence_one_hot = one_hot_encode(total_valence, num_classes=8)
            features.append(total_valence_one_hot)

        # 3. Aromaticity
        if self.config.get('aromatic', False):
            aromatic = torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
            features.append(aromatic)

        # 4. Hybridization One-Hot
        if self.config.get('hybridization', False):
            hybridization = atom.GetHybridization()
            hybridization_idx = self.hybridization_dict.get(
                hybridization, self.hybridization_dict[Chem.HybridizationType.OTHER]
            )
            hybridization_one_hot = one_hot_encode(hybridization_idx, self.num_hybridization)
            features.append(hybridization_one_hot)

        # 5. Formal Charge One-Hot (Shifted to handle negative values)
        if self.config.get('formal_charge', False):
            formal_charge = atom.GetFormalCharge()
            # Shift formal_charge to make it non-negative (assuming range [-2, +2])
            formal_charge_shifted = formal_charge + 2
            # Clamp to [0,4] for one_hot with num_classes=5
            formal_charge_shifted = min(max(formal_charge_shifted, 0), 4)
            formal_charge_one_hot = one_hot_encode(formal_charge_shifted, num_classes=5)
            features.append(formal_charge_one_hot)

        # 6. Default Valence One-Hot
        if self.config.get('default_valence', False):
            default_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
            # Clamp to [0,7] for one_hot with num_classes=8
            default_valence = min(max(default_valence, 0), 7)
            default_valence_one_hot = one_hot_encode(default_valence, num_classes=8)
            features.append(default_valence_one_hot)

        # 7. Ring Sizes
        if self.config.get('ring_size', False):
            ring_sizes = [atom.IsInRingSize(r) for r in range(3, 8)]
            ring_size_tensor = torch.tensor(ring_sizes, dtype=torch.float)
            features.append(ring_size_tensor)

        # 8. Number of Hydrogens One-Hot
        if self.config.get('hydrogen_count', False):
            # Calculate total number of hydrogens attached to the atom
            attached_H = sum([1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1])
            explicit = atom.GetNumExplicitHs()
            implicit = atom.GetNumImplicitHs()
            H_num = attached_H + explicit + implicit
            # Clamp H_num to [0,5] for one_hot with num_classes=6
            H_num = min(max(H_num, 0), 5)
            H_num_one_hot = one_hot_encode(H_num, num_classes=6)
            features.append(H_num_one_hot)

        # Concatenate all features
        try:
            feature_tensor = torch.cat(features, dim=0)
        except RuntimeError as e:
            feature_shapes = [f.shape for f in features]
            raise RuntimeError(
                f"Error concatenating features for atom index {atom.GetIdx()}: {e}\n"
                f"Feature shapes: {feature_shapes}"
            )

        return feature_tensor