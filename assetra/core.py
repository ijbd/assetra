from logging import getLogger
from abc import abstractmethod, ABC
from typing import List

log = getLogger(__name__)

class EnergyUnit(ABC):
	@abstractmethod
	def get_hourly_generation(self):
		'''Return a single time-series'''
		pass

class EnergySystem:
	def __init__(self):
		self._energy_units = []

	def add_energy_unit(self, energy_unit):
		self._energy_units.append(energy_unit)

	def remove_energy_unit(self, energy_unit):
		self._energy_units.remove(energy_unit)

	def get_hourly_generation(self):
		'''Return an array of time-series for all energy_units'''
		for energy_source in self.energy_sources:
			get_hourly_generation(energy_unit)

class ProbabilisticAnalysisModel:
	def __init__(
		self, 
		energy_system: EnergySystem,
		num_monte_carlo_iterations: int,
		num_time_steps: int):
		# initialize state variables
		self.energy_system = energy_system
		self.num_monte_carlo_iterations = num_monte_carlo_iterations
		self.num_time_steps = num_time_steps
		
	def run(self):
		self.hourly_generation_matrix = da.zeros((
			self.num_monte_carlo_iterations,
			self.num_time_steps,
			self.num_generators))
		self.hourly_net_generation_matrix = da.zeros((
			self.num_monte_carlo_iterations,
			self.num_time_steps))

		for it in range(num_monte_carlo_iterations):
			self.hourly_generation_matrix[it] = \
				self.energy_system.get_hourly_generation()
			
		self.hourly_net_generation_matrix = \
			da.sum(self.hourly_net_generation_matrix[iterations])	

class ResourceAdequacyMetric:
	pass

class ResourceContributionMetric:
	pass