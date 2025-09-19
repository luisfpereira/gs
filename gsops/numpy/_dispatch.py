import sys

from gsops import BACKEND_NAME

if BACKEND_NAME == "autograd":
    from autograd import numpy, scipy


else:
    import numpy
    import scipy

    sys.modules[__name__ + ".scipy.linalg"] = scipy.linalg


sys.modules[__name__ + ".numpy"] = numpy
sys.modules[__name__ + ".numpy.linalg"] = numpy.linalg
sys.modules[__name__ + ".scipy"] = scipy
sys.modules[__name__ + ".scipy.special"] = scipy.special
