import functools

import gsops._backend_config as _config

from .._dtype_utils import (
    _dyn_update_dtype,  # noqa: F401
    _modify_func_default_dtype,  # noqa: F401
    _pre_add_default_dtype_by_casting,
    _pre_allow_complex_dtype,
    _pre_cast_fout_to_input_dtype,
    _pre_cast_out_from_dtype,
    _pre_cast_out_to_input_dtype,
    _pre_set_default_dtype,
    get_default_cdtype,  # noqa: F401
    get_default_dtype,  # noqa: F401
)
from ._dispatch import numpy as _np

_DTYPES = {
    _np.dtype("int32"): 0,
    _np.dtype("int64"): 1,
    _np.dtype("float32"): 2,
    _np.dtype("float64"): 3,
    _np.dtype("complex64"): 4,
    _np.dtype("complex128"): 5,
}

_COMPLEX_DTYPES = [
    _np.complex64,
    _np.complex128,
]


def _box_unary_scalar(target=None):
    """Update dtype if input is float in unary operations.

    How it works?
    -------------
    If dtype is float, then default dtype is passed as argument.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):
            if isinstance(x, float):
                return func(x, *args, dtype=_config.DEFAULT_DTYPE, **kwargs)

            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_binary_scalar(target=None):
    """Update dtype if input is float in binary operations.

    How it works?
    -------------
    If dtype is float, then default dtype is passed as argument.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):
            if isinstance(x1, float):
                return func(x1, x2, *args, dtype=_config.DEFAULT_DTYPE, **kwargs)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _dtype_as_str(dtype):
    return dtype.name


def _get_wider_dtype(tensor_list):
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(dtype_list) == 1:
        return dtype_list[0], True

    wider_dtype_index = max(dtype_list)
    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    return wider_dtype, False


def is_floating(x):
    return x.dtype.kind == "f"


def is_complex(x):
    return x.dtype.kind == "c"


def is_bool(x):
    return x.dtype.kind == "b"


def as_dtype(value):
    """Transform string representing dtype in dtype."""
    return _np.dtype(value)


def cast(x, dtype):
    return x.astype(dtype)


set_default_dtype = _pre_set_default_dtype(as_dtype)


def convert_to_wider_dtype(tensor_list):
    wider_dtype, same = _get_wider_dtype(tensor_list)
    if same:
        return tensor_list

    return [cast(x, dtype=wider_dtype) for x in tensor_list]


_add_default_dtype_by_casting = _pre_add_default_dtype_by_casting(cast)
_cast_fout_to_input_dtype = _pre_cast_fout_to_input_dtype(cast, is_floating)
_cast_out_to_input_dtype = _pre_cast_out_to_input_dtype(
    cast, is_floating, is_complex, as_dtype, _dtype_as_str
)

_cast_out_from_dtype = _pre_cast_out_from_dtype(cast, is_floating, is_complex)
_allow_complex_dtype = _pre_allow_complex_dtype(cast, _COMPLEX_DTYPES)
