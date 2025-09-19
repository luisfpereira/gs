"""Autograd based random backend."""

from autograd.numpy.random import randint, seed  # noqa: F401

from ..numpy.random import (  # noqa: F401
    choice,
    multivariate_normal,
    normal,
    rand,
    uniform,
)
