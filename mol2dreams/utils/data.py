import torch
from torch.nn import functional as F
import tqdm
import numpy as np
import pandas as pd

def one_hot_encode(labels, num_classes, allow_unknown=True, unknown_class=None):
    """
    Encodes labels into one-hot vectors.

    Args:
        labels (int or list of int): Labels to encode.
        num_classes (int): Total number of classes.
            - If allow_unknown=True, num_classes should include the 'Unknown' class.
            - For example, if you have 5 known classes and 1 'Unknown' class, set num_classes=6.
        allow_unknown (bool): Whether to allow an 'Unknown' class.
        unknown_class (int, optional): Index for 'Unknown' class. Must be within [0, num_classes-1].
            Defaults to num_classes - 1.

    Returns:
        torch.Tensor: One-hot encoded tensor.
            - If input is a single label, shape: [num_classes]
            - If input is a list of labels, shape: [len(labels), num_classes]
    """
    if isinstance(labels, int):
        labels = [labels]
        single = True
    else:
        single = False

    labels = torch.tensor(labels, dtype=torch.long)

    if allow_unknown:
        if unknown_class is None:
            unknown_class = num_classes - 1
        else:
            if not (0 <= unknown_class < num_classes):
                raise ValueError(f"'unknown_class' must be within [0, {num_classes -1}], got {unknown_class}")

        # Create a mask for labels that are out of range
        unknown_mask = labels >= num_classes
        # Assign 'unknown_class' index to out-of-range labels
        labels_clamped = torch.where(unknown_mask, torch.tensor(unknown_class, device=labels.device), labels)
        # Additionally, clamp to ensure labels are non-negative
        labels_clamped = torch.clamp(labels_clamped, min=0)
    else:
        if (labels < 0).any() or (labels >= num_classes).any():
            raise ValueError("Labels out of range and allow_unknown is False.")
        labels_clamped = labels

    one_hot = F.one_hot(labels_clamped, num_classes=num_classes).float()

    if single:
        one_hot = one_hot.squeeze(0)

    return one_hot


def prepare_datasets(msdata, embs, splits=['train', 'val', 'test'],
                     smiles_col='smiles', embedding_col='DreaMS_embedding', fold_col='FOLD',
                     extra_cols=['COLLISION_ENERGY', 'adduct', 'precursor_mz']):
    """
    Prepares train, validation, and test datasets from MSData and embeddings, ensuring unique IDENTIFIERs.

    Args:
        msdata (MSData): The MSData object loaded from HDF5.
        embs (np.ndarray or torch.Tensor): Embeddings matrix with shape [num_samples, embedding_size].
        splits (list of str): List of fold names to extract. Default is ['train', 'valid', 'test'].
        smiles_col (str): Column name for SMILES strings in the DataFrame. Default is 'smiles'.
        embedding_col (str): Column name for embeddings in the DataFrame. Default is 'DreaMS_embedding'.
        fold_col (str): Column name for fold attribute in the DataFrame. Default is 'FOLD'.
        extra_cols (list of str): Additional columns to include in the datasets.

    Returns:
        dict: A dictionary where keys are split names and values are lists of dictionaries
              with 'smiles', 'embedding', and extra attributes.
    """
    # Convert msdata to pandas DataFrame
    df = msdata.to_pandas()

    # Check alignment
    num_rows = df.shape[0]
    if embs.shape[0] != num_rows:
        raise ValueError(f"Number of embeddings ({embs.shape[0]}) does not match number of data samples ({num_rows}).")

    # Assign embeddings to the DataFrame
    # Ensure embeddings are numpy arrays
    if isinstance(embs, torch.Tensor):
        embs = embs.numpy()
    elif not isinstance(embs, np.ndarray):
        embs = np.array(embs)

    df[embedding_col] = list(embs)

    datasets = {split: [] for split in splits}

    for split in splits:
        split_df = df[df[fold_col] == split].reset_index(drop=True)
        print(f"Processing split '{split}' with {len(split_df)} samples.")

        for idx, row in tqdm.tqdm(split_df.iterrows(), total=split_df.shape[0], desc=f"Featurizing {split}"):
            identifier = row.get('IDENTIFIER', None)
            smiles = row.get(smiles_col, None)
            embedding = row.get(embedding_col, None)

            extra_attrs = {col: row.get(col, None) for col in extra_cols}

            if pd.isna(smiles):
                print(f"Skipping row {idx} due to missing SMILES.")
                continue
            if pd.isna(identifier):
                print(f"Skipping row {idx} due to missing IDENTIFIER.")
                continue
            if embedding is None or len(embedding) != embs.shape[1]:
                print(f"Skipping row {idx} due to invalid embedding.")
                continue

            datasets[split].append({
                'IDENTIFIER': identifier,
                'smiles': smiles,
                'embedding': embedding,
                **extra_attrs
            })

    unique_folds = df[fold_col].unique()
    additional_folds = set(unique_folds) - set(splits)
    for split in additional_folds:
        split_df = df[df[fold_col] == split].reset_index(drop=True)
        print(f"Processing additional split '{split}' with {len(split_df)} samples.")
        datasets[split] = []
        for idx, row in tqdm(split_df.iterrows(), total=split_df.shape[0], desc=f"Featurizing {split}"):
            identifier = row.get('IDENTIFIER', None)
            smiles = row.get(smiles_col, None)
            embedding = row.get(embedding_col, None)

            extra_attrs = {col: row.get(col, None) for col in extra_cols}

            if pd.isna(smiles):
                print(f"Skipping row {idx} in split '{split}' due to missing SMILES.")
                continue
            if pd.isna(identifier):
                print(f"Skipping row {idx} in split '{split}' due to missing IDENTIFIER.")
                continue
            if embedding is None or len(embedding) != embs.shape[1]:
                print(f"Skipping row {idx} in split '{split}' due to invalid embedding.")
                continue

            datasets[split].append({
                'IDENTIFIER': identifier,
                'smiles': smiles,
                'embedding': embedding,
                **extra_attrs
            })

    return datasets
