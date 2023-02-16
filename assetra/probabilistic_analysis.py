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

    def run(self):

        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
        )

        # initialize capacity matrix
        self.hourly_capacity_matrix = xr.DataArray(
            data=np.zeros(
                (self._trial_size, self._energy_system.size, len(time_stamps))
            ),
            coords=dict(
                trial=np.arange(self._trial_size),
                energy_unit=[u.id for u in self._energy_system.energy_units],
                time=time_stamps,
            ),
        )

        # initialize net capacity matrix
        self.net_hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self._trial_size, len(time_stamps))),
            coords=dict(trial=np.arange(self._trial_size), time=time_stamps),
        )

        # simulate resource adequacy
        for trial in range(self._trial_size):
            self.hourly_capacity_matrix.loc[
                trial
            ] = self._energy_system.get_hourly_capacity_by_unit(
                self._start_hour,
                self._end_hour,
                self.net_hourly_capacity_matrix.loc[trial],
            )
