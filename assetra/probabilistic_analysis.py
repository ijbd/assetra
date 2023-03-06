from __future__ import annotations
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List
from logging import getLogger

# package
from assetra.core import (
    EnergySystem,
    EnergySystemBuilder,
    VOLATILE_UNIT_TYPES,
    NONVOLATILE_UNIT_TYPES,
)

# external
import numpy as np
import xarray as xr

log = getLogger(__name__)

MAX_ITERATIONS = 10


class ProbabilisticSimulation:
    """Class responsible for creating/storing the Monte Carlo
    trials for EnergySystem objects."""

    def __init__(self, start_hour: datetime, end_hour: datetime, trial_size: int):
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._trial_size = trial_size

        # state variables
        self._energy_system = None
        self._net_hourly_capacity_matrix = None
        self._hourly_capacity_matrix = None

    def copy(self):
        return ProbabilisticSimulation(
            self._start_hour, self._end_hour, self._trial_size
        )

    def assign_energy_system(self, energy_system: EnergySystem):
        self._energy_system = energy_system
        self._net_hourly_capacity_matrix = None
        self._hourly_capacity_matrix = None

    @property
    def net_hourly_capacity_matrix(self):
        if self._net_hourly_capacity_matrix is None:
            self.run()
        return self._net_hourly_capacity_matrix.copy()

    def get_hourly_capacity_matrix_by_type(self, unit_type):
        if self._hourly_capacity_matrix is None:
            self.run()
        return self._hourly_capacity_matrix.sel(unit_type=unit_type)

    def run(self, net_hourly_capacity_matrix: xr.DataArray = None):

        # check for energy system
        if not isinstance(self._energy_system, EnergySystem):
            raise RuntimeError(
                "Energy system not assigned to probabilistic simulation object."
            )

        # setup net hourly capacity matrix
        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
        )

        # initialize net capacity matrix
        if net_hourly_capacity_matrix is not None:
            # check dimensions
            assert net_hourly_capacity_matrix.sizes["trial"] == self._trial_size
            assert net_hourly_capacity_matrix.sizes["time"] == time_stamps.size
            assert net_hourly_capacity_matrix.time[0] == time_stamps[0]
            # load net hourly capacity
            self._net_hourly_capacity_matrix = net_hourly_capacity_matrix
        else:
            self._net_hourly_capacity_matrix = xr.DataArray(
                data=np.zeros((self._trial_size, len(time_stamps))),
                coords=dict(trial=np.arange(self._trial_size), time=time_stamps),
            )

        # initialize capacity by unit type
        unit_types = list(self._energy_system.unit_datasets)
        self._hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((len(unit_types), self._trial_size, len(time_stamps))),
            coords=dict(
                unit_type=unit_types,
                trial=np.arange(self._trial_size),
                time=time_stamps,
            ),
        )

        # iterate through unit datasets
        for unit_type, unit_dataset in self._energy_system.unit_datasets.items():
            self._hourly_capacity_matrix.loc[
                unit_type
            ] = unit_type.get_probabilistic_capacity_matrix(
                unit_dataset,
                self._net_hourly_capacity_matrix,
            ).values
            self._net_hourly_capacity_matrix += self._hourly_capacity_matrix.sel(
                unit_type=unit_type
            ).values


class ResourceAdequacyMetric(ABC):
    def __init__(self, simulation: ProbabilisticSimulation):
        self.simulation = simulation

    @abstractmethod
    def evaluate(self, simulation):
        pass


class ExpectedUnservedEnergy(ResourceAdequacyMetric):
    def evaluate(self):
        hourly_unserved_energy = self.simulation.net_hourly_capacity_matrix.where(
            self.simulation.net_hourly_capacity_matrix < 0, 0
        )
        return float(
            -hourly_unserved_energy.sum() / hourly_unserved_energy.sizes["trial"]
        )


class LossOfLoadHours(ResourceAdequacyMetric):
    def evaluate(self):
        hourly_outages = self.simulation.net_hourly_capacity_matrix < 0
        return float(hourly_outages.sum() / hourly_outages.sizes["trial"])


class TransmissionProbabilisticSimulation:
    # TODO
    pass


class TransmissionSystem:
    # TODO
    pass


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

        # decompose system into volatile and non-volatile components
        # non-volatile simulation
        self._original_system_non_volatile = self._original_system.get_system_by_type(
            NONVOLATILE_UNIT_TYPES
        )
        self._original_non_volatile_simulation = self._simulation.copy()
        self._original_non_volatile_simulation.assign_energy_system(
            self._original_system_non_volatile
        )

        # volatile simulation
        self._original_system_volatile = self._original_system.get_system_by_type(
            VOLATILE_UNIT_TYPES
        )
        self._original_volatile_simulation = self._simulation.copy()
        self._original_volatile_simulation.assign_energy_system(
            self._original_system_volatile
        )

        # run chained simulation
        self._original_non_volatile_simulation.run()
        self._original_volatile_simulation.run(
            self._original_non_volatile_simulation.net_hourly_capacity_matrix
        )
        self._original_resource_adequacy = self._resource_adequacy_metric(
            self._original_volatile_simulation
        ).evaluate()
        print(self._original_resource_adequacy)

    def evaluate(self, addition: EnergySystem, threshold=0.001):
        # need three systems
        """
        To efficiently compute ELCC, we can consider four separate systems whose probabilistic simulations build on one anothers:
            (1) a system with the original static units
            (2) a system with the original static and *volatile units
            (3) a system with the original *volatile units and the *new static and *volatile units
            (4) a system with the original static and *volatile units and the new static and *volatile units.
        Given that we can add the net capacity of one simulation to another, and that we will need to run the probabilistic simulation several times, we should construct intermediate energy systems:
            (1) a system with the original static and volatile units (evaluated once in the init)
            (2) a system with the original volatile units and the new static and volatile units (evaluated once here)
            (3) a system with the original and new volatile units (evaluated iteratively to find elcc)
        """
        if self._original_resource_adequacy == 0:
            log.error("Invalid ELCC calculation for system with no shortfalls.")
            raise RuntimeError()

        # decompose system into volatile and non-volatile components
        # non-volatile simulation
        additional_system_non_volatile = addition.get_system_by_type(
            NONVOLATILE_UNIT_TYPES
        )
        additional_non_volatile_simulation = self._simulation.copy()
        additional_non_volatile_simulation.assign_energy_system(
            additional_system_non_volatile
        )

        # volatile simulation
        additional_system_volatile = addition.get_system_by_type(VOLATILE_UNIT_TYPES)
        additional_volatile_simulation = self._simulation.copy()
        additional_volatile_simulation.assign_energy_system(additional_system_volatile)

        # run non-volatile_simulation
        additional_non_volatile_simulation.run()

        # get non-volatile net hourly capacity
        non_volatile_net_hourly_capacity_matrix = (
            self._original_non_volatile_simulation.net_hourly_capacity_matrix
            + additional_non_volatile_simulation.net_hourly_capacity_matrix
        )

        # add load
        additional_demand_upper_bound = addition.nameplate_capacity
        additional_demand_lower_bound = 0
        additional_demand = (
            additional_demand_lower_bound
            + (additional_demand_upper_bound - additional_demand_lower_bound) / 2
        )

        # run chained volatile simulation
        self._original_volatile_simulation.run(
            non_volatile_net_hourly_capacity_matrix - additional_demand
        )
        additional_volatile_simulation.run(
            self._original_volatile_simulation.net_hourly_capacity_matrix
        )

        # update resource adequacy
        new_resource_adequacy_model = self._resource_adequacy_metric(
            additional_volatile_simulation
        )
        new_resource_adequacy = new_resource_adequacy_model.evaluate()
        diff = abs(new_resource_adequacy - self._original_resource_adequacy)

        # iterate until convergence
        iteration = 0

        while diff > threshold:
            print(iteration, self._original_resource_adequacy, new_resource_adequacy)
            # check iteration count
            if iteration > MAX_ITERATIONS:
                return additional_demand

            # iterate until original resource adequacy level is met
            if new_resource_adequacy > self._original_resource_adequacy:
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

            # run chained volatile simulation
            self._original_volatile_simulation.run(
                non_volatile_net_hourly_capacity_matrix - additional_demand
            )
            additional_volatile_simulation.run(
                self._original_volatile_simulation.net_hourly_capacity_matrix
            )

            # update resource adequacy
            new_resource_adequacy = new_resource_adequacy_model.evaluate()
            diff = abs(new_resource_adequacy - self._original_resource_adequacy)

            # update iteration count
            iteration += 1

        return float(additional_demand)
