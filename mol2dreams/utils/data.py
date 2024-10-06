import torch
from torch.nn import functional as F

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
