"""Autograd based random backend."""

from autograd.numpy.random import randint, seed

from ..numpy.random import choice, multivariate_normal, normal, rand, uniform
