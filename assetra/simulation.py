from __future__ import annotations
from datetime import datetime
from logging import getLogger

# external
import numpy as np
import xarray as xr

# package
from assetra.system import EnergySystem

log = getLogger(__name__)

class ProbabilisticSimulation:
    """Class responsible for creating/storing the Monte Carlo
    trials for EnergySystem objects."""

    def __init__(
        self, start_hour: datetime, end_hour: datetime, trial_size: int
    ):
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
            log.warning(
                "Energy system not assigned to simulation object."
                )
            raise RuntimeError()

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
                coords=dict(
                    trial=np.arange(self._trial_size), time=time_stamps
                ),
            )

        # initialize capacity by unit type
        unit_types = list(self._energy_system.unit_datasets)
        self._hourly_capacity_matrix = xr.DataArray(
            data=np.zeros(
                (len(unit_types), self._trial_size, len(time_stamps))
            ),
            coords=dict(
                unit_type=unit_types,
                trial=np.arange(self._trial_size),
                time=time_stamps,
            ),
        )

        # iterate through unit datasets
        for (
            unit_type,
            unit_dataset,
        ) in self._energy_system.unit_datasets.items():
            self._hourly_capacity_matrix.loc[
                unit_type
            ] = unit_type.get_probabilistic_capacity_matrix(
                unit_dataset,
                self._net_hourly_capacity_matrix,
            ).values
            self._net_hourly_capacity_matrix += (
                self._hourly_capacity_matrix.sel(unit_type=unit_type).values
            )

