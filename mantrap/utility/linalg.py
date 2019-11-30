import numpy as np
import torch


def normalize(v):
    """Normalize vector v using L2 norm. If norm is 0 then return vector, else normalized vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def normalize_torch(v):
    """Normalize vector v using L2 norm. If norm is 0 then return vector, else normalized vector (torch tensors)."""
    norm = torch.norm(v)
    return torch.div(v, norm) if norm > 1e-6 else v
