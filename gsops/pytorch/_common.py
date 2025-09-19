import torch as _torch


def from_numpy(x):
    if _torch.is_tensor(x):
        return x
    return _torch.from_numpy(x)


def array(val, dtype=None):
    return _torch.asarray(val, dtype=dtype)
