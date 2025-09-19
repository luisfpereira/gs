"""Numpy based computation backend."""

try:
    import torch as _torch
except ImportError:
    pass

from .._backend_config import np_atol as atol  # noqa: F401
from .._backend_config import np_rtol as rtol  # noqa: F401
from . import (  # noqa: F401
    autodiff,
    linalg,
    random,
    sparse,
)
from ._dispatch import BACKEND_NAME
from ._dispatch import numpy as _np
from ._dispatch import scipy as _scipy
from ._dispatch.numpy import (  # noqa: F401
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
from ._dispatch.scipy.special import (  # noqa: F401
    erf,
    gamma,
    polygamma,
)
from ._dtype import (  # noqa: F401
    _box_binary_scalar,
    _box_unary_scalar,
    _cast_out_from_dtype,
    _dyn_update_dtype,
    _get_wider_dtype,
    _modify_func_default_dtype,
    as_dtype,
    cast,
    convert_to_wider_dtype,
    get_default_cdtype,
    get_default_dtype,
    is_bool,
    is_complex,
    is_floating,
    set_default_dtype,
)

try:
    from ._dispatch.numpy import trapezoid
except ImportError:
    from ._dispatch.numpy import trapz as trapezoid  # noqa: F401


if BACKEND_NAME != "autograd":
    ones = _modify_func_default_dtype(target=_np.ones)
    eye = _modify_func_default_dtype(target=_np.eye)

linspace = _dyn_update_dtype(target=_np.linspace, dtype_pos=5)
empty = _dyn_update_dtype(target=_np.empty, dtype_pos=1)
array = _cast_out_from_dtype(target=_np.array, dtype_pos=1)
zeros = _dyn_update_dtype(target=_np.zeros, dtype_pos=1)

abs = _box_unary_scalar(target=_np.abs)
arccos = _box_unary_scalar(target=_np.arccos)
arccosh = _box_unary_scalar(target=_np.arccosh)
arcsin = _box_unary_scalar(target=_np.arcsin)
arctanh = _box_unary_scalar(target=_np.arctanh)
ceil = _box_unary_scalar(target=_np.ceil)
cos = _box_unary_scalar(target=_np.cos)
cosh = _box_unary_scalar(target=_np.cosh)
exp = _box_unary_scalar(target=_np.exp)
floor = _box_unary_scalar(target=_np.floor)
log = _box_unary_scalar(target=_np.log)
sign = _box_unary_scalar(target=_np.sign)
sin = _box_unary_scalar(target=_np.sin)
sinh = _box_unary_scalar(target=_np.sinh)
sqrt = _box_unary_scalar(target=_np.sqrt)
tan = _box_unary_scalar(target=_np.tan)
tanh = _box_unary_scalar(target=_np.tanh)

arctan2 = _box_binary_scalar(target=_np.arctan2)
mod = _box_binary_scalar(target=_np.mod)
power = _box_binary_scalar(target=_np.power)


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return _is_boolean(x[0])
    if isinstance(x, _np.ndarray):
        return x.dtype == bool
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if isinstance(x, _np.ndarray):
        return x.ndim > 0
    return False


def is_array(x):
    return type(x) is _np.ndarray


def to_ndarray(x, to_ndim, axis=0, dtype=None):
    x = _np.asarray(x, dtype=dtype)

    if x.ndim > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")

    while x.ndim < to_ndim:
        x = _np.expand_dims(x, axis=axis)

    return x


def angle(z, deg=False):
    out = _np.angle(z, deg=deg)
    if isinstance(z, float):
        return cast(out, get_default_dtype())

    return out


def imag(x):
    out = _np.imag(x)
    if is_array(x):
        return out

    return get_default_dtype().type(out)


def real(x):
    out = _np.real(x)
    if is_array(x):
        return out

    return array(out)


def arange(start_or_stop, /, stop=None, step=1, dtype=None, **kwargs):
    if dtype is None and (
        isinstance(stop, float)
        or isinstance(step, float)
        or isinstance(start_or_stop, float)
    ):
        dtype = get_default_dtype()

    if stop is None:
        return _np.arange(start_or_stop, step=step, dtype=dtype)

    return _np.arange(start_or_stop, stop, step=step, dtype=dtype)


def to_numpy(x):
    return x


def from_numpy(x):
    return x


def squeeze(x, axis=None):
    if axis is None:
        return _np.squeeze(x)
    if x.shape[axis] != 1:
        return x
    return _np.squeeze(x, axis=axis)


def flatten(x):
    return x.flatten()


def one_hot(labels, num_classes):
    return eye(num_classes, dtype=_np.dtype("uint8"))[labels]


def assignment(x, values, indices, axis=0):
    """Assign values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)

    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] = values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    len_indices = len(indices) if _is_iterable(indices) else 1
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        if not zip_indices:
            len_indices = len(indices) if _is_iterable(indices) else 1
        len_values = len(values) if _is_iterable(values) else 1
        if len_values > 1 and len_values != len_indices:
            raise ValueError("Either one value or as many values as indices")
        x_new[indices] = values
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] = values
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    values : {float, list(float)}
        Value or list of values to be assigned.
    indices : {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)

    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] += values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_indices = len(indices) if _is_iterable(indices) else 1
        len_values = len(values) if _is_iterable(values) else 1
        if len_values > 1 and len_values != len_indices:
            raise ValueError("Either one value or as many values as indices")
        x_new[indices] += values
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] += values
    return x_new


def ndim(x):
    return x.ndim


def get_slice(x, indices):
    """Return a slice of an array, following Numpy's style.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    indices : iterable(iterable(int))
        Indices which are kept along each axis, starting from 0.

    Returns
    -------
    slice : array-like
        Slice of x given by indices.

    Notes
    -----
    This follows Numpy's convention: indices are grouped by axis.

    Examples
    --------
    >>> a = np.array(range(30)).reshape(3,10)
    >>> get_slice(a, ((0, 2), (8, 9)))
    array([8, 29])
    """
    return x[indices]


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


def set_diag(x, new_diag):
    """Set the diagonal along the last two axis.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    new_diag : array-like, shape=[dim[-2]]
        Values to set on the diagonal.

    Returns
    -------
    None

    Notes
    -----
    This mimics tensorflow.linalg.set_diag(x, new_diag), when new_diag is a
    1-D array, but modifies x instead of creating a copy.
    """
    arr_shape = x.shape
    x[..., range(arr_shape[-2]), range(arr_shape[-1])] = new_diag
    return x


def copy(x):
    return x.copy()


def array_from_sparse(indices, data, target_shape):
    """Create an array of given shape, with values at specific indices.

    The rest of the array will be filled with zeros.

    Parameters
    ----------
    indices : iterable(tuple(int))
        Index of each element which will be assigned a specific value.
    data : iterable(scalar)
        Value associated at each index.
    target_shape : tuple(int)
        Shape of the output array.

    Returns
    -------
    a : array, shape=target_shape
        Array of zeros with specified values assigned to specified indices.
    """
    data = array(data)
    out = zeros(target_shape, dtype=data.dtype)
    out.put(_np.ravel_multi_index(_np.array(indices).T, target_shape), data)
    return out


def vec_to_diag(vec):
    """Convert vector to diagonal matrix."""
    d = vec.shape[-1]
    return _np.squeeze(vec[..., None, :] * _np.eye(d, dtype=vec.dtype)[None, :, :])


def tril_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = _np.tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = _np.triu_indices(n, k=k)
    return x[..., rows, cols]


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    """Build matrix from given components.

    Forms a matrix from diagonal, strictly upper triangular and
    strictly lower traingular parts.

    Parameters
    ----------
    diag : array_like, shape=[..., n]
    tri_upp : array_like, shape=[..., (n * (n - 1)) / 2]
    tri_low : array_like, shape=[..., (n * (n - 1)) / 2]

    Returns
    -------
    mat : array_like, shape=[..., n, n]
    """
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])

    n = diag.shape[-1]
    (i,) = _np.diag_indices(n, ndim=1)
    j, k = _np.triu_indices(n, k=1)
    mat = zeros(diag.shape + (n,), dtype=diag.dtype)
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def divide(a, b, ignore_div_zero=False):
    if ignore_div_zero is False:
        return _np.divide(a, b)

    wider_dtype, _ = _get_wider_dtype([a, b])
    return _np.divide(a, b, out=zeros(a.shape, dtype=wider_dtype), where=b != 0)


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _np.ravel_multi_index(idxs, size)


def matmul(*args, **kwargs):
    for arg in args:
        if arg.ndim == 1:
            raise ValueError("ndims must be >=2")
    return _np.matmul(*args, **kwargs)


def outer(a, b):
    if a.ndim > 1 and b.ndim > 1:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.multiply.outer(a, b)
    if b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out


def matvec(A, b):
    if b.ndim == 1:
        return _np.matmul(A, b)
    if A.ndim == 2:
        return _np.matmul(A, b.T).T
    return _np.einsum("...ij,...j->...i", A, b)


def dot(a, b):
    if b.ndim == 1:
        return _np.dot(a, b)

    if a.ndim == 1:
        return _np.dot(a, b.T)

    return _np.einsum("...i,...i->...", a, b)


def trace(a):
    return _np.trace(a, axis1=-2, axis2=-1)


def scatter_add(input, dim, index, src):
    """Add values from src into input at the indices specified in index.

    Parameters
    ----------
    input : array-like
        Tensor to scatter values into.
    dim : int
        The axis along which to index.
    index : array-like
        The indices of elements to scatter.
    src : array-like
        The source element(s) to scatter.

    Returns
    -------
    input : array-like
        Modified input array.
    """

    if dim == 0:
        for i, val in zip(index, src):
            input[i] += val
        return input
    if dim == 1:
        for j in range(len(input)):
            for i, val in zip(index[j], src[j]):
                # TODO: make this one particular
                if not isinstance(val, _np.float64) and BACKEND_NAME == "autograd":
                    val = float(val._value)
                input[j, i] += float(val)
        return input
    raise NotImplementedError


def scatter_sum_1d(index, src, size=None):
    shape_ = None if size is None else (size, 1)

    dummy_indices = _np.zeros_like(index)

    return _np.array(
        _scipy.sparse.coo_matrix(
            (src, (index, dummy_indices)),
            shape=shape_,
        ).todense()
    ).flatten()


def to_device(a, device):
    return a


def argsort(a, axis=-1):
    return _np.argsort(a, axis=axis)


def to_torch(a):
    return _torch.tensor(a)


def diag(array):
    return _np.diag(array)


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return False
