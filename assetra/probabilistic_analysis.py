
# package
from assetra.core import EnergySystem, BulkEnergySystem

# external
import numpy as np

class ProbabilisticSimulation:
	'''Class responsible for creating/storing the Monte Carlo
	trials for EnergySystem objects.'''
	def __init__(
		self, 
		energy_system: EnergySystem,
		start_hour: int,
		end_hour: int,
		trial_size: int):
		self._energy_system = energy_system
		self._start_hour = start_hour
		self._end_hour = end_hour
		self._trial_size = trial_size

	@property
	def hours(self):
		return self._end_hour - self._start_hour
		
	def run(self):
		self.hourly_capacity_matrix = np.zeros((self._trial_size, 
			self._energy_system.size, self._end_hour - self._start_hour))
		for trial in range(self._trial_size):
			self.hourly_capacity_matrix[trial] = \
				self._energy_system.get_hourly_capacity_by_unit(
					self._start_hour, self._end_hour)

class BulkProbabilisticSimulation:
	'''Class responsible for creating/storing the Monte Carlo
	trials for BulkEnergySystem objects.'''
	def __init__(
		self,
		bulk_energy_system: BulkEnergySystem,
		start_hour: int,
		end_hour: int,
		trial_size: int):
		pass

	def run(self):
		pass
