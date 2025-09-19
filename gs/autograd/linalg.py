"""Autograd based linear algebra backend."""

import functools as _functools

import autograd.numpy as _np
from autograd.extend import defvjp as _defvjp
from autograd.extend import primitive as _primitive
from autograd.numpy.linalg import (  # noqa: F401
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_power,
    matrix_rank,
    norm,
    svd,
)
from autograd.scipy.linalg import expm

from ..numpy.linalg import (  # noqa: F401
    fractional_matrix_power,
    is_single_matrix_pd,
    polar,
    qr,
    quadratic_assignment,
    solve,
    solve_sylvester,
    sqrtm,
)
from ..numpy.linalg import logm as _logm


def _adjoint(_ans, x, fn):
    vectorized = x.ndim == 3
    axes = (0, 2, 1) if vectorized else (1, 0)

    def vjp(g):
        n = x.shape[-1]
        size_m = x.shape[:-2] + (2 * n, 2 * n)
        # TODO: fix dtype here?
        mat = _np.zeros(size_m)
        mat[..., :n, :n] = x.transpose(axes)
        mat[..., n:, n:] = x.transpose(axes)
        mat[..., :n, n:] = g
        return fn(mat)[..., :n, n:]

    return vjp


_expm_vjp = _functools.partial(_adjoint, fn=expm)
_defvjp(expm, _expm_vjp)


logm = _primitive(_logm)

_logm_vjp = _functools.partial(_adjoint, fn=logm)
_defvjp(logm, _logm_vjp)
