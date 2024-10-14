from torch.nn import functional as F
import torch
from torch_geometric.data import DataLoader

def cosine_similarity_metric(outputs, targets):

    outputs_normalized = F.normalize(outputs, p=2, dim=1)
    targets_normalized = F.normalize(targets, p=2, dim=1)

    cosine_sim_normalized = F.cosine_similarity(outputs_normalized, targets_normalized, dim=1).mean().item()


    cosine_sim = F.cosine_similarity(outputs, targets, dim=1).mean().item()
    return cosine_sim, cosine_sim_normalized

def compute_cosine_similarity(model, data_path, device='cpu', batch_size=32, num_workers=0):
    """
    Computes the average cosine similarity between model predictions and ground truth labels.

    Args:
        model (torch.nn.Module): Trained Mol2DreaMS model.
        data_path (str): Path to the precomputed validation data file (e.g., '../../data/data/precomputed_global_features_batches_simple_val.pt').
        device (str, optional): Device to run the computations on ('cpu' or 'cuda'). Defaults to 'cpu'.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.

    Returns:
        float: Average cosine similarity over the dataset.
    """
    # Load the validation data
    data_list = torch.load(data_path)  # List of Data objects

    # Create a DataLoader for batching
    data_loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Move the model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    total_cosine_sim = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Move batch to the device
            batch = batch.to(device)

            # Forward pass through the model
            outputs = model(batch)  # Shape: [batch_size, 1024]
            targets = batch.y.to(device)  # Shape: [batch_size, 1024]

            # Compute cosine similarity for each sample in the batch
            # Resulting shape: [batch_size]
            cosine_sim = F.cosine_similarity(outputs, targets, dim=1)

            # Accumulate the sum of cosine similarities
            total_cosine_sim += cosine_sim.sum().item()
            total_samples += batch.num_graphs

    # Calculate the average cosine similarity
    average_cosine_sim = total_cosine_sim / total_samples if total_samples > 0 else 0.0

    return average_cosine_sim