from abc import abstractmethod, ABC

# package
from assetra.probabilistic_analysis import ProbabilisticSimulation

# external
import numpy as np


class ResourceAdequacyMetric(ABC):
    """Class responsible for evaluating resource adequacy of
    a ProbabilisticSimulation object."""

    def __init__(self, probabilistic_simulation: ProbabilisticSimulation):
        self._simulation = probabilistic_simulation

    @abstractmethod
    def evaluate(self):
        pass


class ExpectedUnservedEnergy(ResourceAdequacyMetric):
    def evaluate(self):
        hourly_unserved_energy = self._simulation.net_hourly_capacity_matrix.where(
            self._simulation.net_hourly_capacity_matrix < 0, 0
        )
        return float(
            -hourly_unserved_energy.sum() / hourly_unserved_energy.sizes["trial"]
        )


class LossOfLoadHours(ResourceAdequacyMetric):
    def evaluate(self):
        pass


class LossOfLoadDays(ResourceAdequacyMetric):
    def evaluate(self):
        pass
