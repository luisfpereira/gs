from geomstats.test.parametrizers import DataBasedParametrizer

from .cases import AgainstScipyTestCase
from .data.against_scipy import AgainstScipyTestData


class TestAgainstScipy(AgainstScipyTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstScipyTestData()
