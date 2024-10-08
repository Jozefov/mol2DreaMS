from torch.nn import functional as F

def cosine_similarity_metric(outputs, targets):
    outputs_norm = F.normalize(outputs, p=2, dim=1)
    targets_norm = F.normalize(targets, p=2, dim=1)
    cosine_sim = (outputs_norm * targets_norm).sum(dim=1).mean()
    return cosine_sim.item()