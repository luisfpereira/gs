try:
    import torch as _torch
except ImportError:
    pass

from ._dispatch import scipy as _scipy


def to_dense(array):
    return array.todense()


def from_scipy_coo(array):
    return array


def from_scipy_csc(array):
    return array


def from_scipy_csr(array):
    return array


def from_scipy_dia(array):
    return array


def to_scipy_csc(array):
    return array


def to_scipy_dia(array):
    return array


def coo_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    if not coalesce:
        return _scipy.sparse.coo_matrix((values, indices), shape=shape, dtype=dtype)

    return _scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=dtype).tocoo()


def to_coo(array):
    return array.tocoo()


def to_csc(array):
    return array.tocsc()


def to_csr(array):
    return array.tocsr()


def to_torch_csc(array):
    values = _torch.from_numpy(array.data)
    ccol_indices = _torch.from_numpy(array.indptr)
    row_indices = _torch.from_numpy(array.indices)

    return _torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=array.shape)


def to_torch_dia(array):
    return to_torch_csc(array.tocsc())


def to_torch_coo(array):
    indices = _torch.stack([_torch.from_numpy(array.row), _torch.from_numpy(array.col)])
    values = _torch.from_numpy(array.data)

    return _torch.sparse_coo_tensor(indices, values, array.shape)


def csr_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    # TODO: need to adapt to other arg combinations
    # TODO: expose other arguments
    if not coalesce:
        return _scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=dtype)

    return _scipy.sparse.coo_matrix((values, indices), shape=shape, dtype=dtype).tocsr()


def csc_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    if not coalesce:
        return _scipy.sparse.csc_matrix((values, indices), shape=shape, dtype=dtype)

    return _scipy.sparse.coo_matrix((values, indices), shape=shape, dtype=dtype).tocsc()


def dia_matrix(diagonals, offsets=0, shape=None, dtype=None):
    if shape is None:
        shape = (diagonals.shape[-1], diagonals.shape[-1])
    return _scipy.sparse.dia_matrix((diagonals, offsets), shape=shape)
