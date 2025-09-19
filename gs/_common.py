import math as _math

from numpy import pi  # noqa: F401


def comb(n, k):
    return _math.factorial(n) // _math.factorial(k) // _math.factorial(n - k)
