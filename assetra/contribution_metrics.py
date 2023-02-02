from abc import abstractmethod, ABC

# package
from assetra.core import EnergySystem
from assetra.probabilistic_analysis import ProbabilisticSimulation


class ResourceContributionMetric(ABC):
    """Class responsible for evaluating the resource contribution
    of one energy system object to another."""

    def __init__(self, resource_adequacy_model, addition: EnergySystem):
        self._resource_adequacy_model = resource_adequacy_model
        self._addition = addition

    @abstractmethod
    def evaluate():
        pass


class EffectiveLoadCarryingCapability(ResourceContributionMetric):
    def __init__(
        self,
        probabilistic_simulation: ProbabilisticSimulation,
        addition: EnergySystem,
        threshold: float,
    ):
        ResourceContributionMetric.__init__(
            self, probabilistic_simulation, addition
        )
        self._threshold = threshold

    def get_resource_contribution(self):

        # find original resource adeqaucy
        original_adequacy = self._resource_adequacy_model.evaluate()

        # add new resources
        self._resource_adequacy_model.add_system(self._addition)

        # add load
        additional_demand_upper_bound = self._addition.capacity
        additional_demand_lower_bound = 0
        additional_demand = (
            additional_demand_lower_bound
            + (additional_demand_upper_bound - additional_demand_lower_bound)
            / 2
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

            # update constant demand
            self._resource_adequacy.set_demand_offset(additional_demand)

            # update resource adequacy
            new_adequacy = self._resource_adequacy_model.evaluate()
            diff = abs(new_adequacy - original_adequacy)

        # remove new resources
        self._resource_adequacy_model.remove_system(self._addition)
