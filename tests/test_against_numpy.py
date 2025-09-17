from geomstats.test.parametrizers import DataBasedParametrizer

from .cases import AgainstNumpyTestCase
from .data.against_numpy import AgainstNumpyTestData


class TestAgainstNumpy(AgainstNumpyTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstNumpyTestData()
