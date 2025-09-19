import sys

from gs import BACKEND_NAME

if BACKEND_NAME == "autograd":
    from autograd import numpy, scipy

    from ..autograd import _common

else:
    import numpy
    import scipy

    from ..numpy import _common

    sys.modules[__name__ + ".scipy.linalg"] = scipy.linalg


sys.modules[__name__ + ".numpy"] = numpy
sys.modules[__name__ + ".numpy.linalg"] = numpy.linalg
sys.modules[__name__ + ".scipy"] = scipy
sys.modules[__name__ + ".scipy.special"] = scipy.special
sys.modules[__name__ + "._common"] = _common
