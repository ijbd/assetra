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
    """Derived ResourceAdequacyMetric class responsible for calculating loss of
    load hours:

    "[The] expected count of event-hours per study horizon."

    https://gridops.epri.com/Adequacy/metrics#Loss_of_Load_Hours_.28LOLH.29

    Args:
        simulation (ProbabilisticSimulation) : Simulation object to
            characterize.
    """

    def evaluate(self) -> float:
        """Return loss of load hours

        Returns:
            float: LOLH in units of hours
        """
        hourly_outages = self.simulation.net_hourly_capacity_matrix < 0
        return float(hourly_outages.sum() / hourly_outages.sizes["trial"])


class LossOfLoadDays(ResourceAdequacyMetric):
    """Derived ResourceAdequacyMetric class responsible for calculating loss of
    load days:

    "[The] expected count of event-days per study horizon."

    https://gridops.epri.com/Adequacy/metrics#Loss_of_Load_Days_.28LOLD.2C_LOLEd.2Fyr.29

    Args:
        simulation (ProbabilisticSimulation) : Simulation object to
            characterize.
    """

    def evaluate(self) -> float:
        """Return loss of load days

        Returns:
            float: LOLD in units of days
        """
        hourly_outages = self.simulation.net_hourly_capacity_matrix < 0
        daily_outages = hourly_outages.resample(time="1D").max()
        return float(daily_outages.sum() / hourly_outages.sizes["trial"])


class LossOfLoadFrequency(ResourceAdequacyMetric):
    """Derived ResourceAdequacyMetric class responsible for calculating loss of
    load frequency:

    "[The] expected count of adequacy events per study horizon, with an
    adequacy event defined as a contiguous set of hours with a shortfall

    https://gridops.epri.com/Adequacy/metrics#Loss_of_Load_Events_.28LOLEv.2C_LOLF.29

    Args:
        simulation (ProbabilisticSimulation) : Simulation object to
            characterize.
    """

    def evaluate(self) -> float:
        """Return loss of load frequency

        Returns:
            float: LOLF in units of # of events
        """
        hourly_outages = (
            self.simulation.net_hourly_capacity_matrix < 0
        ).astype(int)
        shifted_hourly_outages = hourly_outages.roll(time=1)
        hourly_outage_transitions = (
            hourly_outages - shifted_hourly_outages
        ) > 0
        return float(
            (hourly_outage_transitions.sum() + hourly_outages[:, -1].sum())
            / hourly_outages.sizes["trial"]
        )
