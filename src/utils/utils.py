import torch


def logmatmulexp(a, b):
    if b.dim() == 1:
        b = b.unsqueeze(1)

    a_pl_b_expanded = a.unsqueeze(2) + b.unsqueeze(0)

    result = torch.logsumexp(a_pl_b_expanded, dim=1)

    return result.squeeze()
