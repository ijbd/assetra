from abc import ABC, abstractmethod

from assetra.simulation import ProbabilisticSimulation


class ResourceAdequacyMetric(ABC):
    def __init__(self, simulation: ProbabilisticSimulation):
        self.simulation = simulation

    @abstractmethod
    def evaluate(self, simulation):
        pass


class ExpectedUnservedEnergy(ResourceAdequacyMetric):
    def evaluate(self) -> float:
        hourly_unserved_energy = (
            self.simulation.net_hourly_capacity_matrix.where(
                self.simulation.net_hourly_capacity_matrix < 0, 0
            )
        )
        return float(
            -hourly_unserved_energy.sum()
            / hourly_unserved_energy.sizes["trial"]
        )


class LossOfLoadHours(ResourceAdequacyMetric):
    def evaluate(self) -> float:
        hourly_outages = self.simulation.net_hourly_capacity_matrix < 0
        return float(hourly_outages.sum() / hourly_outages.sizes["trial"])
