from geomstats.test.parametrizers import DataBasedParametrizer

from .cases import AgainstEinsumTestCase
from .data.against_einsum import AgainstEinsumTestData


class TestAgainstEinsum(AgainstEinsumTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstEinsumTestData()
