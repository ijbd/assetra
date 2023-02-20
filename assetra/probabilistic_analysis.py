from datetime import datetime

# package
from assetra.energy_system import EnergySystem

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

        # setup net hourly capacity matrix
        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
        )
        
        # initialize net capacity matrix
        self.net_hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self._trial_size, len(time_stamps))),
            coords=dict(trial=np.arange(self._trial_size), time=time_stamps),
        )

        # aggregate static units 
        net_hourly_capacity_matrix += static_units['hourly capacity'].sum(dim='energy_unit')

        # sample outages for stochastic units
        net_hourly_capacity_matrix += stochastic_units['hourly capacity'].where(
            np.random((self._trial_size, stochastic_units.sizes['energy unit'], stochastic_units.sizes['time']))
            > stochastic_units['hourly_forced_outage_rates']
            ).sum(dim='energy_unit')

        # sequential operation
        for unit in sequential_units:
            for net_hourly_capacity in net_hourly_capacity_matrix:
                net_hourly_capacity += unit.get_hourly_capacity(net_hourly_capacity)

        '''
        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
        )

         # initialize net capacity matrix
        self.net_hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self._trial_size, len(time_stamps))),
            coords=dict(trial=np.arange(self._trial_size), time=time_stamps),
        )
        # initialize capacity matrix
        self.hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self._trial_size, self._energy_system.size, len(time_stamps))),
            coords=dict(
                trial=np.arange(self._trial_size),
                energy_unit=[u.id for u in self._energy_system.energy_units],
                time=time_stamps,
            ),
        )
        '''
