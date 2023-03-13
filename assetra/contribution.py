from assetra.units import RESPONSIVE_UNIT_TYPES, NONRESPONSIVE_UNIT_TYPES
from assetra.system import EnergySystem
from assetra.simulation import ProbabilisticSimulation
from assetra.metrics import ResourceAdequacyMetric

MAX_ITERATIONS = 10

class EffectiveLoadCarryingCapability:
    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
    ):
        self._original_system = energy_system
        self._simulation = simulation
        self._resource_adequacy_metric = resource_adequacy_metric

        # decompose system into responsive and non-responsive components
        # non-responsive simulation
        self._original_system_non_responsive = (
            self._original_system.get_system_by_type(NONRESPONSIVE_UNIT_TYPES)
        )
        self._original_non_responsive_simulation = self._simulation.copy()
        self._original_non_responsive_simulation.assign_energy_system(
            self._original_system_non_responsive
        )

        # responsive simulation
        self._original_system_responsive = (
            self._original_system.get_system_by_type(RESPONSIVE_UNIT_TYPES)
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
        self._original_resource_adequacy = self._resource_adequacy_metric(
            self._original_responsive_simulation
        ).evaluate()

    def evaluate(self, addition: EnergySystem, threshold=0.001):
        # need three systems
        """
        To efficiently compute ELCC, we can consider four separate systems whose probabilistic simulations build on one anothers:
            (1) a system with the original static units
            (2) a system with the original static and *responsive units
            (3) a system with the original *responsive units and the *new static and *responsive units
            (4) a system with the original static and *responsive units and the new static and *responsive units.
        Given that we can add the net capacity of one simulation to another, and that we will need to run the probabilistic simulation several times, we should construct intermediate energy systems:
            (1) a system with the original static and responsive units (evaluated once in the init)
            (2) a system with the original responsive units and the new static and responsive units (evaluated once here)
            (3) a system with the original and new responsive units (evaluated iteratively to find elcc)
        """
        if self._original_resource_adequacy == 0:
            log.error("Invalid ELCC calculation for system with no shortfalls.")
            raise RuntimeError()

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
        additional_demand = (
            additional_demand_lower_bound
            + (additional_demand_upper_bound - additional_demand_lower_bound)
            / 2
        )

        # run chained responsive simulation
        self._original_responsive_simulation.run(
            non_responsive_net_hourly_capacity_matrix - additional_demand
        )
        additional_responsive_simulation.run(
            self._original_responsive_simulation.net_hourly_capacity_matrix
        )

        # update resource adequacy
        new_resource_adequacy_model = self._resource_adequacy_metric(
            additional_responsive_simulation
        )
        new_resource_adequacy = new_resource_adequacy_model.evaluate()
        diff = abs(new_resource_adequacy - self._original_resource_adequacy)

        # iterate until convergence
        iteration = 0

        while diff > threshold:
            # check iteration count
            if iteration > MAX_ITERATIONS:
                return additional_demand

            # iterate until original resource adequacy level is met
            if new_resource_adequacy > self._original_resource_adequacy:
                # if over-reliable, add load
                additional_demand_upper_bound = additional_demand
                additional_demand = (
                    additional_demand_lower_bound
                    + (
                        additional_demand_upper_bound
                        - additional_demand_lower_bound
                    )
                    / 2
                )
            else:
                # if under-reliable, remove load
                additional_demand_lower_bound = additional_demand
                additional_demand = (
                    additional_demand_lower_bound
                    + (
                        additional_demand_upper_bound
                        - additional_demand_lower_bound
                    )
                    / 2
                )

            # run chained responsive simulation
            self._original_responsive_simulation.run(
                non_responsive_net_hourly_capacity_matrix - additional_demand
            )
            additional_responsive_simulation.run(
                self._original_responsive_simulation.net_hourly_capacity_matrix
            )

            # update resource adequacy
            new_resource_adequacy = new_resource_adequacy_model.evaluate()
            diff = abs(new_resource_adequacy - self._original_resource_adequacy)

            # update iteration count
            iteration += 1

        return float(additional_demand)
