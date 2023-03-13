from abc import ABC, abstractmethod

from assetra.simulation import ProbabilisticSimulation


class ResourceAdequacyMetric(ABC):
    """Class responsible for quantifying resource adequacy of a
    ProbabilisticSimulation object

    Args:
        simulation (ProbabilisticSimulation) : Simulation object to
            characterize.
    """

    def __init__(self, simulation: ProbabilisticSimulation):
        self.simulation = simulation

    @abstractmethod
    def evaluate(self) -> float:
        """Return resource adequacy of initialized simulation.

        Returns:
            float: Evaluated resource adequacy metric.
        """
        pass


class ExpectedUnservedEnergy(ResourceAdequacyMetric):
    """Derived ResourceAdequacyMetric class responsible for calculating
    expected unserved energy:

    "[The] total expected amount of unserved energy ... in a given study
    horizon."

    https://gridops.epri.com/Adequacy/metrics#Magnitude_Metrics

    Args:
        simulation (ProbabilisticSimulation) : Simulation object to
            characterize.
    """

    def evaluate(self) -> float:
        """Return expected unserved energy

        Returns:
            float: EUE in units of energy
        """
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
    """Derived ResourceAdequacyMetric class responsible for calculating
    loss of load hours:

    "[The] expected count of event-hours per study horizon."

    https://gridops.epri.com/Adequacy/metrics#Loss_of_Load_Hours_.28LOLH.29

    Args:
        simulation (ProbabilisticSimulation) : Simulation object to
            characterize.
    """

    def evaluate(self) -> float:
        """Return loss of load hours

        Returns:
            float: LOLH in units of time
        """
        hourly_outages = self.simulation.net_hourly_capacity_matrix < 0
        return float(hourly_outages.sum() / hourly_outages.sizes["trial"])
