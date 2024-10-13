import torch
from mol2dreams.global_constants import (
    COLLISION_ENERGY_BINS,
    COLLISION_ENERGY_NUM_BINS,
    ADDUCT_CATEGORIES,
    INSTRUMENT_TYPE_CATEGORIES,
    ADDUCT_NUM_CLASSES,
    INSTRUMENT_TYPE_NUM_CLASSES,
    PRECURSOR_MZ_MAX
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import bisect


class GlobalFeaturizer:
    def __init__(self, config):
        """
        Initializes the GlobalFeaturizer with specified configurations.

        Args:
            config (dict): Configuration dictionary specifying which global features to extract.
                           Example:
                           {
                               'features': {
                                   'collision_energy': True,
                                   'adduct': True,
                                   'instrument_type': True,
                                   'precursor_mz': True
                               }
                           }
        """
        self.config = config.get('features', {})

        # Define categorical features and their possible categories
        self.categorical_features = {}
        if self.config.get('adduct', False):
            self.categorical_features['adduct'] = ADDUCT_CATEGORIES
        if self.config.get('instrument_type', False):
            self.categorical_features['instrument_type'] = INSTRUMENT_TYPE_CATEGORIES

        # Initialize LabelEncoders for categorical features
        self.label_encoders = {}
        for feature, categories in self.categorical_features.items():
            # Add 'unknown' category for unseen values
            categories_with_unknown = categories + ['unknown']
            le = LabelEncoder()
            le.fit(categories_with_unknown)
            self.label_encoders[feature] = le

        # Initialize Binning for collision_energy if enabled
        self.collision_energy_bins = None
        if self.config.get('collision_energy', False):
            self.collision_energy_bins = COLLISION_ENERGY_BINS  # [0,10,20,...,100]
            self.collision_energy_num_bins = COLLISION_ENERGY_NUM_BINS  # 10

    def bin_collision_energy(self, value):
        """
        Assigns a collision energy value to a bin index.

        Args:
            value (float): Collision energy value.

        Returns:
            int: Bin index (0-based).
        """
        # Cap the value at the maximum bin edge
        if value >= self.collision_energy_bins[-1]:
            return self.collision_energy_num_bins - 1  # Last bin
        # Find the right bin
        bin_index = bisect.bisect_right(self.collision_energy_bins, value) - 1
        return bin_index

    def one_hot_encode(self, index, num_classes):
        """
        One-hot encodes an index.

        Args:
            index (int): Index to encode.
            num_classes (int): Total number of classes.

        Returns:
            torch.Tensor: One-hot encoded tensor of shape [num_classes].
        """
        one_hot = torch.zeros(num_classes, dtype=torch.float)
        if 0 <= index < num_classes:
            one_hot[index] = 1.0
        else:
            # Handle out-of-range indices by setting all to zero or another strategy
            pass
        return one_hot

    def featurize(self, extra_attrs):
        """
        Extracts and encodes global features from the extra_attrs dictionary.

        Args:
            extra_attrs (dict): Dictionary containing global attributes.

        Returns:
            torch.Tensor: Concatenated tensor of all global features.
                            Shape: [batch_size, num_encoded_features]
        """
        feature_list = []

        # Process collision_energy as binned categorical
        if self.config.get('collision_energy', False):
            value = extra_attrs.get('collision_energy', 0.0)
            value_capped = min(float(value), self.collision_energy_bins[-1])  # Cap at 100
            bin_index = self.bin_collision_energy(value_capped)
            # One-hot encode the bin index
            collision_energy_one_hot = self.one_hot_encode(bin_index, self.collision_energy_num_bins)  # Shape: [num_bins]
            feature_list.append(collision_energy_one_hot)  # [num_bins]

        # Process precursor_mz as continuous (capped)
        if self.config.get('precursor_mz', False):
            value = extra_attrs.get('precursor_mz', 0.0)
            value_capped = min(float(value), PRECURSOR_MZ_MAX)
            value_tensor = torch.tensor([value_capped], dtype=torch.float)  # Shape: [1]
            feature_list.append(value_tensor)  # [1]

        # Process categorical features with One-Hot Encoding
        for feature, categories in self.categorical_features.items():
            value = extra_attrs.get(feature, 'unknown')
            if value not in categories:
                value = 'unknown'
            le = self.label_encoders[feature]
            try:
                encoded = le.transform([value])[0]
            except ValueError:
                # If the value is unseen, encode as 'unknown'
                encoded = le.transform(['unknown'])[0]
            # One-hot encode the category
            if feature == 'adduct':
                num_classes = ADDUCT_NUM_CLASSES  # e.g., 3
            elif feature == 'instrument_type':
                num_classes = INSTRUMENT_TYPE_NUM_CLASSES  # e.g., 3
            else:
                num_classes = le.classes_.shape[0]
            one_hot = self.one_hot_encode(encoded, num_classes)  # Shape: [num_classes]
            feature_list.append(one_hot)  # [num_classes]

        # Concatenate all features
        if feature_list:
            global_features = torch.cat(feature_list, dim=0)  # [num_encoded_features]
            global_features = global_features.unsqueeze(0)  # Shape: [1, num_encoded_features]
        else:
            global_features = torch.tensor([], dtype=torch.float).unsqueeze(0)  # Empty tensor if no features

        return global_features  # Shape: [1, num_encoded_features]