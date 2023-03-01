from datetime import datetime

# package
from assetra.core import EnergySystem

# external
import numpy as np
import xarray as xr


class ProbabilisticSimulation:
    """Class responsible for creating/storing the Monte Carlo
    trials for EnergySystem objects."""

    def __init__(
        self,
        energy_system: EnergySystem,
        start_hour: datetime,
        end_hour: datetime,
        trial_size: int,
    ):
        self._energy_system = energy_system
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._trial_size = trial_size

        self._net_hourly_capacity_matrix = None
        self._hourly_capacity_matrix = None

    @property
    def net_hourly_capacity_matrix(self):
        if self._net_hourly_capacity_matrix is None:
            self.run()
        return self._net_hourly_capacity_matrix

    def get_capacity_matrix_by_type(self, unit_type: type):
        if self._hourly_capacity_matrix is None:
            self.run()
        return self._hourly_capacity_matrix.loc[:, unit_type]

    def run(self):
        # TODO add loading point for pre-computed fleet capacities
        # setup net hourly capacity matrix
        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
        )

        # initialize capacity by unit type
        unit_types = list(self._energy_system.unit_datasets)
        self._hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self._trial_size, len(unit_types), len(time_stamps))),
            coords=dict(
                trial=np.arange(self._trial_size),
                unit_type=unit_types,
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
                :, unit_type
            ] = unit_type.get_probabilistic_capacity_matrix(
                unit_dataset,
                self._net_hourly_capacity_matrix,
            )
            self._net_hourly_capacity_matrix += self._hourly_capacity_matrix.sel(
                unit_type=unit_type
            ).values


def get_effective_unserved_energy(net_hourly_capacity_matrix):
    pass


def get_loss_of_load_hours(net_hourly_capacity_matrix):
    pass


def get_loss_of_load_days(net_hourly_capacity_matrix):
    pass
