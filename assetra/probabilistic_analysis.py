# package
from assetra.core import EnergySystem, EnergyUnit

# external
import numpy as np


class ProbabilisticSimulation:
    """Class responsible for creating/storing the Monte Carlo
    trials for EnergySystem objects."""

    def __init__(
        self,
        energy_system: EnergySystem,
        start_hour: int,
        end_hour: int,
        trial_size: int
    ):
        self._energy_system = energy_system
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._trial_size = trial_size

        # setup capacity matrix
        self.hourly_capacity_matrix = np.zeros(
            (
                self._trial_size,
                self._energy_system.size,
                self._end_hour - self._start_hour,
            )
        )

        # look up table for energy units
        self._look_up_table = {self._energy_system:0}

    def add_system(self, energy_system:EnergySystem):
        # add system to lookup table
        self._look_up_table[energy_system] = self.hourly_capacity_matrix.shape[1]

        # add system to capacity matrix
        added_hourly_capacity_matrix = np.zeros((self._trial_size, energy_system.size, self._end_hour - self._start_hour))
        self.hourly_capacity_matrix = np.concatenate((self.hourly_capacity_matrix, added_hourly_capacity_matrix), axis=1)

    def remove_system(self, energy_system:EnergySystem):
        # get index of system units
        start_index = self._look_up_table[energy_system]
        system_units_index = np.arange(start_index, start_index + energy_system.size)

        # remove system from lookup table
        self._look_up_table.pop(energy_system)

        # remove system from capacity matrix
        self.hourly_capacity_matrix = np.delete(self.hourly_capacity_matrix, system_units_index, axis=1)

    @property
    def hours(self):
        return self._end_hour - self._start_hour

    def run(self):
        
        for trial in range(self._trial_size):
            self.hourly_capacity_matrix[
                trial
            ] = self._energy_system.get_hourly_capacity_by_unit(
                self._start_hour, self._end_hour
            )

        print(self.hourly_capacity_matrix)
