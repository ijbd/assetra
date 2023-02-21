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
        self.energy_system = energy_system
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.trial_size = trial_size

    def run(self):

        # setup net hourly capacity matrix
        time_stamps = xr.date_range(
            self.start_hour, self.end_hour, freq="H", inclusive="both"
        )
        
        # initialize net capacity matrix
        self.net_hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self.trial_size, len(time_stamps))),
            coords=dict(trial=np.arange(self.trial_size), time=time_stamps),
        )

        # get capacity by unit type
        unit_types = list(self.energy_system.unit_datasets)
        self.hourly_capacity_matrix = xr.DataArray(
            data=np.zeros((self.trial_size, len(unit_types), len(time_stamps))),
            coords=dict(trial=np.arange(self.trial_size), unit_type=unit_types, time=time_stamps)
        )

        # iterate through unit datasets
        for unit_type, unit_dataset in self.energy_system.unit_datasets.items():
            self.hourly_capacity_matrix.loc[:, unit_type] = unit_type.get_probabilistic_capacity_matrix(unit_dataset, self.start_hour, self.end_hour, self.trial_size, self.net_hourly_capacity_matrix)
            self.net_hourly_capacity_matrix += self.hourly_capacity_matrix.loc[:, unit_type]
            
                

