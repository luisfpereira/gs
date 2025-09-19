"""Numpy based random backend."""

from ._dispatch import numpy as _np
from ._dispatch.numpy.random import default_rng as _default_rng  # noqa: F401
from ._dispatch.numpy.random import randint, seed  # noqa: F401
from ._dtype import (
    _allow_complex_dtype,
    _modify_func_default_dtype,
)

rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.rand)
)

uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.uniform)
)


normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.normal)
)

multivariate_normal = _modify_func_default_dtype(
    copy=False,
    kw_only=True,
    target=_allow_complex_dtype(target=_np.random.multivariate_normal),
)


def choice(*args, **kwargs):
    return _np.random.default_rng().choice(*args, **kwargs)
