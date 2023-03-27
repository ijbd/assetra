from logging import getLogger
from abc import ABC, abstractmethod

from assetra.units import RESPONSIVE_UNIT_TYPES, NONRESPONSIVE_UNIT_TYPES
from assetra.system import EnergySystem
from assetra.simulation import ProbabilisticSimulation
from assetra.metrics import ResourceAdequacyMetric

import xarray as xr

LOG = getLogger(__name__)


class ResourceContributionMetric(ABC):
    """Class responsible for quantifying resource contributions
    to an energy system

    Args:
        energy_system (EnergySystem): Base system to which resources are added
        simulation (ProbabilisticSimulation): Instantiated simulation object
            whose parameters (i.e. time range and trial size are used
            throughout the ELCC calculation)
        resource_adequacy_metric: Class derived
            from ResourceAdequacyMetric to use throughout the ELCC simulation
            (e.g. ExpectedUnservedEnergy, LossOfLoadHours)
    """

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
    ):
        self._original_energy_system = energy_system
        self._simulation = simulation
        self._resource_adequacy_metric = resource_adequacy_metric

    @abstractmethod
    def evaluate(self, addition: EnergySystem) -> float:
        """Return resource contribution of addition to the energy system

        Args:
            addition (EnergySystem): Energy system to add (possibly a single
            unit)

        Returns:
            float: Quantified resource contribution of addition to the energy
                system.
        """


class EffectiveLoadCarryingCapability(ResourceContributionMetric):
    """Class responsible for quantifying resource contribution to an energy
    system using the effective load-carrying capability metric:

    "[The] amount by which the system's load can increase when the resource is
    added to the system while maintaining the same system adequacy."

    https://gridops.epri.com/Adequacy/metrics#Effective_Load_Carrying_Capability_.28ELCC.29

    Args:
        energy_system (EnergySystem) : Base system to which resources are added
        simulation (ProbabilisticSimulation) : Instantiated simulation object
            whose parameters (i.e. time range and trial size are used
            throughout the ELCC calculation)
        resource_adequacy_metric (type[ResourceAdequacyMetric') : Class derived
            from ResourceAdequacyMetric to use throughout the ELCC simulation
            (e.g. ExpectedUnservedEnergy, LossOfLoadHours)
    """

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
    ):
        """The ELCC computes resource adequacy many times iteratively. In many
        instances, it is neither necessary nor desirable to re-run the
        probabilistic simulation for a set of units more than once. For
        so-called non-responsive units (static and stochastic units) which do
        not respond to system conditions, a single probabilistic simulation
        is sufficient. For responsive units, including storage, changing system
        conditions necessitate re-evaluation of probabilistic capacity
        contributions.

        To account for this classification in the iterative resource adequacy
        calculations, the ELCC decomposes energy systems into their responsive
        and non-responsive components. In each iteration, combinations of the
        component energy systems' probabilistic simulations are "chained"
        together depending on which need to be recomputed.

        For example, to establish the baseline resource adequacy (as in the
        init function), both the original responsive system and the
        original non-responsive systems are evaluated. Programmatically, the
        non-responsive system is evaluated first. Then the resulting
        net hourly capacity matrix is fed into the responsive system.

        When a new system is added, it is again decomposed into responsive and
        non-responsive subsystems. The original non-responsive system does not
        need to be re-evaluated, so its already-simulated probabilistic net
        hourly capacity is fed into the new non-responsive system which is fed
        into the new and original responsive systems respectively.

        In this way, non-responsive units are only evaluated once while
        responsive units are able to respond to changes in system conditions.
        """
        ResourceContributionMetric.__init__(
            self, energy_system, simulation, resource_adequacy_metric
        )

        # decompose system into responsive and non-responsive components
        # non-responsive simulation
        self._original_system_non_responsive = (
            self._original_energy_system.get_system_by_type(
                NONRESPONSIVE_UNIT_TYPES
            )
        )
        self._original_non_responsive_simulation = self._simulation.copy()
        self._original_non_responsive_simulation.assign_energy_system(
            self._original_system_non_responsive
        )

        # responsive simulation
        self._original_system_responsive = (
            self._original_energy_system.get_system_by_type(
                RESPONSIVE_UNIT_TYPES
            )
        )
        self._original_responsive_simulation = self._simulation.copy()
        self._original_responsive_simulation.assign_energy_system(
            self._original_system_responsive
        )

        # run chained simulation
        self._original_non_responsive_simulation.run()
        self._original_responsive_simulation.run(
            self._original_non_responsive_simulation.net_hourly_capacity_matrix
        )
        self._resource_adequacy_model = self._resource_adequacy_metric(
            self._original_responsive_simulation
        )
        self._original_resource_adequacy = (
            self._resource_adequacy_model.evaluate()
        )
        LOG.info(
            "Original resource adequacy: "
            + str(self._original_resource_adequacy)
        )

        if self._original_resource_adequacy == 0:
            LOG.error("Invalid ELCC calculation for system with no risk")
            raise RuntimeWarning()

        # save intermediate steps
        self._original_net_capacity_matrix = (
            self._original_responsive_simulation.net_hourly_capacity_matrix
        )
        self._intermediate_net_capacity_matrices = []

    @property
    def original_net_capacity_matrix(self) -> xr.DataArray:
        """Return the net hourly capacity matrix of the base system"""
        # TODO test
        return self._original_net_capacity_matrix.copy()

    @property
    def intermediate_net_capacity_matrices(
        self,
    ) -> tuple[tuple[float, xr.DataArray]]:
        """Return intermediate net capacity matrices for each step of the ELCC
        calculation. Each element is a tuple composed of the amount of added
        constant load and the net hourly capacity matrix corresponding to that
        step.
        """
        # TODO test
        return tuple(m for m in self._intermediate_net_capacity_matrices)

    def evaluate(
        self,
        addition: EnergySystem,
        additional_demand_resolution_pct: float = 0.01,
    ) -> float:
        """Return the ELCC of an addition to the energy system.

        Args:
            addition (EnergySystem): Energy system to add (possibly a
                single unit).
            additional_demand_resolution_pct (float, optional): Resolution of
                added demand to find as percent of added nameplate capacity.
                Defaults to 0.01 (1%). E.g. for a 100 MW addition, the ELCC
                will be found within 1 MW.

        Returns:
            float: Amount of added constant load in units of power.
        """

        # reset intermediate net capacities
        self._intermediate_net_capacity_matrices = []

        # decompose system into responsive and non-responsive components
        # non-responsive simulation
        additional_system_non_responsive = addition.get_system_by_type(
            NONRESPONSIVE_UNIT_TYPES
        )
        additional_non_responsive_simulation = self._simulation.copy()
        additional_non_responsive_simulation.assign_energy_system(
            additional_system_non_responsive
        )

        # responsive simulation
        additional_system_responsive = addition.get_system_by_type(
            RESPONSIVE_UNIT_TYPES
        )
        additional_responsive_simulation = self._simulation.copy()
        additional_responsive_simulation.assign_energy_system(
            additional_system_responsive
        )

        # run non-responsive_simulation
        additional_non_responsive_simulation.run()

        # get non-responsive net hourly capacity
        non_responsive_net_hourly_capacity_matrix = (
            self._original_non_responsive_simulation.net_hourly_capacity_matrix
            + additional_non_responsive_simulation.net_hourly_capacity_matrix
        )

        # add load
        additional_demand_upper_bound = addition.system_capacity
        additional_demand_lower_bound = 0
        additional_demand_resolution = (
            additional_demand_upper_bound - additional_demand_lower_bound
        )
        additional_demand = (
            additional_demand_lower_bound + additional_demand_resolution / 2
        )

        # run chained responsive simulation
        additional_responsive_simulation.run(
            non_responsive_net_hourly_capacity_matrix - additional_demand
        )
        self._original_responsive_simulation.run(
            additional_responsive_simulation.net_hourly_capacity_matrix
        )
        self._intermediate_net_capacity_matrices.append(
            (
                additional_demand,
                self._original_responsive_simulation.net_hourly_capacity_matrix,
            )
        )

        # update resource adequacy
        new_resource_adequacy = self._resource_adequacy_model.evaluate()

        # printout
        LOG.info("Additional demand: " + str(round(additional_demand)))
        LOG.info("Resource adequacy: " + str(new_resource_adequacy))

        # iterate until convergence
        iteration = 0

        while (
            additional_demand_resolution / addition.system_capacity
        ) > additional_demand_resolution_pct:
            # iterate until original resource adequacy level is met
            if new_resource_adequacy > self._original_resource_adequacy:
                # if over-reliable, add load
                additional_demand_upper_bound = additional_demand
            else:
                # if under-reliable, remove load
                additional_demand_lower_bound = additional_demand
            # add demand
            additional_demand_resolution = (
                additional_demand_upper_bound - additional_demand_lower_bound
            )
            additional_demand = (
                additional_demand_lower_bound + additional_demand_resolution / 2
            )

            # run chained responsive simulation
            additional_responsive_simulation.run(
                non_responsive_net_hourly_capacity_matrix - additional_demand
            )
            self._original_responsive_simulation.run(
                additional_responsive_simulation.net_hourly_capacity_matrix
            )
            self._intermediate_net_capacity_matrices.append(
                (
                    additional_demand,
                    self._original_responsive_simulation.net_hourly_capacity_matrix,
                )
            )

            # update resource adequacy
            new_resource_adequacy = self._resource_adequacy_model.evaluate()

            LOG.info("Additional demand: " + str(additional_demand))
            LOG.info("Resource adequacy: " + str(new_resource_adequacy))

            # update iteration count
            iteration += 1

        return float(additional_demand)
