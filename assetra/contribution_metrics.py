from abc import abstractmethod, ABC

# package
from assetra.energy_system import EnergySystem
from assetra.probabilistic_analysis import ProbabilisticSimulation
from assetra.adequacy_metrics import ResourceAdequacyMetric


class EffectiveLoadCarryingCapability:
    def __init__(
        self,
        target: EnergySystem,
        addition: EnergySystem,
        probabilistic_simulation: ProbabilisticSimulation,
        resource_adequacy_metric: ResourceAdequacyMetric,
        threshold: float,
    ):
        self._target = target
        self._addition = addition
        self._probabilistic_simulation = probabilistic_simulation
        self._resource_adequacy_metric = resource_adequacy_metric
        self._addition = addition
        self._threshold = threshold

    def evaluate(self):

        # find original resource adeqaucy
        original_adequacy = self._resource_adequacy_model.evaluate()

        # add new resources
        self._resource_adequacy_model.add_system(self._addition)

        # add load
        additional_demand_upper_bound = self._addition.capacity
        additional_demand_lower_bound = 0
        additional_demand = (
            additional_demand_lower_bound
            + (additional_demand_upper_bound - additional_demand_lower_bound) / 2
        )
        self._resource_adequacy_model.set_demand_offset(additional_demand)

        # update resource adequacy
        new_adequacy = self._resource_adequacy_model.evaluate()
        diff = abs(new_adequacy - original_adequacy)

        while diff > self.threshold:
            # iterate until original resource adequacy level is met
            if self._resource_adequacy_model.evaluate() > original_adequacy:
                # if over-reliable, add load
                additional_demand_upper_bound = additional_demand
                additional_demand = (
                    additional_demand_lower_bound
                    + (additional_demand_upper_bound - additional_demand_lower_bound)
                    / 2
                )

            else:
                # if under-reliable, remove load
                additional_demand_lower_bound = additional_demand
                additional_demand = (
                    additional_demand_lower_bound
                    + (additional_demand_upper_bound - additional_demand_lower_bound)
                    / 2
                )

            # update constant demand
            self._resource_adequacy.set_demand_offset(additional_demand)

            # update resource adequacy
            new_adequacy = self._resource_adequacy_model.evaluate()
            diff = abs(new_adequacy - original_adequacy)

        # remove new resources
        self._resource_adequacy_model.remove_system(self._addition)
