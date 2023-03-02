from datetime import datetime
from abc import ABC, abstractmethod

# package
from assetra.core import EnergySystem

# external
import numpy as np
import xarray as xr


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

    def assign_energy_system(self, energy_system: EnergySystem):
        self._energy_system = energy_system
        self._net_hourly_capacity_matrix = None
        self._hourly_capacity_matrix = None

    @property
    def net_hourly_capacity_matrix(self):
        if self._net_hourly_capacity_matrix is None:
            self.run()
        return self._net_hourly_capacity_matrix

    def get_hourly_capacity_matrix_by_type(self, unit_type):
        if self._hourly_capacity_matrix is None:
            self.run()
        return self._hourly_capacity_matrix.sel(unit_type=unit_type)

    def run(self):
        if not isinstance(self._energy_system, EnergySystem):
            raise RuntimeError(
                "Energy system not assigned to probabilistic simulation object."
            )

        # TODO add loading point for pre-computed fleet capacities
        # setup net hourly capacity matrix
        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
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

        # initialize net capacity matrix
        self._net_hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self._trial_size, len(time_stamps))),
            coords=dict(trial=np.arange(self._trial_size), time=time_stamps),
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
    def __init__(self, simulation):
        self.simulation = simulation

    @abstractmethod
    def evaluate(self):
        pass


class ExpectedUnservedEnergy(ResourceAdequacyMetric):
    def evaluate(self):
        hourly_unserved_energy = self.simulation.net_hourly_capacity_matrix.where(
            self.simulation.net_hourly_capacity_matrix < 0, 0
        )
        return float(
            -hourly_unserved_energy.sum() / hourly_unserved_energy.sizes["trial"]
        )


class TransmissionProbabilisticSimulation:
    # TODO
    pass


class TransmissionSystem:
    # TODO
    pass


class EffectiveLoadCarryingCapability:
    # TODO
    pass
