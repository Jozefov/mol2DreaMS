from torch.nn import functional as F

def cosine_similarity_metric(outputs, targets):

    outputs_normalized = F.normalize(outputs, p=2, dim=1)
    targets_normalized = F.normalize(targets, p=2, dim=1)

    cosine_sim_normalized = F.cosine_similarity(outputs_normalized, targets_normalized, dim=1).mean().item()


    cosine_sim = F.cosine_similarity(outputs, targets, dim=1).mean().item()
    return cosine_sim, cosine_sim_normalized
