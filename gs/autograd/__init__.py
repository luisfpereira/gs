"""Autograd based computation backend."""

import autograd.numpy as _np
from autograd.numpy import (  # noqa: F401
    all,
    allclose,
    amax,
    amin,
    any,
    argmax,
    argmin,
    asarray,
    broadcast_arrays,
    broadcast_to,
    clip,
    complex64,
    complex128,
    concatenate,
    conj,
    cross,
    cumprod,
    cumsum,
    diag_indices,
    diagonal,
    einsum,
    empty_like,
    equal,
    expand_dims,
    flip,
    float32,
    float64,
    geomspace,
    greater,
    hsplit,
    hstack,
    int32,
    int64,
    isclose,
    isnan,
    kron,
    less,
    less_equal,
    logical_and,
    logical_or,
    maximum,
    mean,
    meshgrid,
    minimum,
    moveaxis,
    ones_like,
    pad,
    prod,
    quantile,
    repeat,
    reshape,
    searchsorted,
    shape,
    sort,
    split,
    square,
    stack,
    std,
    sum,
    take,
    tile,
    transpose,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    where,
    zeros_like,
)
from autograd.scipy.special import erf, gamma, polygamma  # noqa: F401

from ..numpy import (  # noqa: F401
    abs,
    angle,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    argsort,
    array,
    array_from_sparse,
    as_dtype,
    assignment,
    assignment_by_sum,
    atol,
    cast,
    ceil,
    convert_to_wider_dtype,
    cos,
    cosh,
    diag,
    divide,
    dot,
    empty,
    exp,
    flatten,
    floor,
    from_numpy,
    get_default_cdtype,
    get_default_dtype,
    get_slice,
    is_array,
    is_bool,
    is_complex,
    is_floating,
    linspace,
    log,
    mat_from_diag_triu_tril,
    matmul,
    matvec,
    mod,
    ndim,
    one_hot,
    power,
    ravel_tril_indices,
    real,
    rtol,
    scatter_add,
    scatter_sum_1d,
    set_default_dtype,
    set_diag,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    tan,
    tanh,
    to_device,
    to_ndarray,
    to_numpy,
    to_torch,
    trace,
    tril_to_vec,
    triu_to_vec,
    vec_to_diag,
    vectorize,
    zeros,
)
from ..numpy._dtype import _dyn_update_dtype
from . import (  # noqa: F401
    autodiff,
    linalg,
    random,
    sparse,
)

try:
    from autograd.numpy import trapezoid
except ImportError:
    from autograd.numpy import trapz as trapezoid  # noqa: F401

ones = _dyn_update_dtype(target=_np.ones)
eye = _dyn_update_dtype(target=_np.eye)


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def imag(x):
    out = _np.imag(x)
    if is_array(x):
        return out

    return array(out)


def copy(x):
    return _np.array(x, copy=True)


def outer(a, b):
    if a.ndim > 1 and b.ndim > 1:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.outer(a, b).reshape(a.shape + b.shape)
    if b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out
