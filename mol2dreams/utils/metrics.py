from torch.nn import functional as F

def cosine_similarity_metric(outputs, targets):

    cosine_sim = F.cosine_similarity(outputs, targets, dim=1).mean().item()
    return cosine_sim