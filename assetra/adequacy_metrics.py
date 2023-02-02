from abc import abstractmethod, ABC

# package
from assetra.probabilistic_analysis import ProbabilisticSimulation

# external
import numpy as np


class ResourceAdequacyMetric(ABC):
    """Class responsible for evaluating resource adequacy of
    a ProbabilisticSimulation object."""

    def __init__(self, probabilistic_simulation: ProbabilisticSimulation):
        self._probabilistic_simulation = probabilistic_simulation

    @property
    def hours(self):
        self._probabilistic_simulation.hours

    @abstractmethod
    def evaluate(self):
        pass

class ExpectedUnservedEnergy(ResourceAdequacyMetric):
    def evaluate(self):
        net_hourly_capacity_by_trial = np.sum(
            self._probabilistic_simulation.hourly_capacity_matrix, axis=1
        )
        hourly_unserved_energy = np.where(
            net_hourly_capacity_by_trial < 0, -net_hourly_capacity_by_trial, 0
        )
        return np.sum(hourly_unserved_energy / net_hourly_capacity_by_trial.size)


class LossOfLoadHours(ResourceAdequacyMetric):
    def evaluate(self):
        pass
