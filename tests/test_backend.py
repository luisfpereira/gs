from geomstats.test.parametrizers import DataBasedParametrizer

from .cases import BackendTestCase
from .data.backend import BackendTestData


class TestBackend(BackendTestCase, metaclass=DataBasedParametrizer):
    testing_data = BackendTestData()
