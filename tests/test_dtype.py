from geomstats.test.parametrizers import DataBasedParametrizer

from .cases import DtypeTestCase
from .data.dtype import DtypeTestData


class TestDtype(DtypeTestCase, metaclass=DataBasedParametrizer):
    testing_data = DtypeTestData()
