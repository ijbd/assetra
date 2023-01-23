from abc import abstractmethod, ABC
from logging import getLogger
from typing import List

# external libraries
from numpy.typing import ArrayLike
import numpy as np

log = getLogger(__name__)

### ENERGY UNIT(S)

class EnergyUnit(ABC):
	def __init__(
		self,
		name: str,
		nameplate_capacity: float):
		self._name = name
		self._nameplate_capacity = nameplate_capacity

	@abstractmethod
	def get_hourly_capacity(self):
		'''Returns a single instance of the hourly capacity of the 
		generating unit.'''
		pass

class StochasticUnit:
	'''Class responsible for returning capacity profile of 
	stochastically-sampled units (i.e. generators).'''
	def __init__(
		self,
		name: str,
		nameplate_capacity: float,
		hourly_capacity: ArrayLike,
		hourly_forced_outage_rate: ArrayLike
		):
		EnergyUnit.__init__(self, name, nameplate_capacity)
		self._hourly_capacity = hourly_capacity
		self._hourly_forced_outage_rate = hourly_forced_outage_rate

	def get_hourly_capacity(self):
		hourly_outage_samples = np.random.random_sample(
			len(self._hourly_capacity)
			)
		hourly_capacity_instance = np.where(
			hourly_outage_samples > self._hourly_forced_outage_rate,
			self._hourly_capacity,
			0)
		return hourly_capacity_instance

class DemandUnit:
	'''Class responsible for returning capacity profile of 
	fixed demand units (i.e. system loads).'''
	def __init__(
		self,
		name: str,
		hourly_demand: ArrayLike
		):
		EnergyUnit.__init__(self, name, 0)
		self._hourly_demand = hourly_demand

	def get_hourly_capacity(self):
		return -(self._hourly_demand)

### ENERGY SYSTEM

class EnergySystem:
	'''Class responsible for managing energy units.'''
	def __init__(
		self,
		name: str):
		self.energy_units = []

	def add_unit(self, energy_unit):
		self.energy_units.append(energy_unit)

	def remove_unit(self, energy_unit):
		self.energy_units.remove(energy_unit)

	def get_hourly_capacity_by_unit(self):
		'''Returns the hourly capacity of each generating unit
		in the energy system.'''
		hourly_capacity_matrix = np.zeros(
			(self.num_generators, 8760))
		hourly_net_capacity = np.zeros(8760)
		for i, energy_unit in enumerate(self.energy_units):
			hourly_capacity_matrix[i] = energy_unit.get_hourly_capacity()
			hourly_net_capacity += hourly_capacity_matrix[i]

		return hourly_capacity_matrix

### BULK ENERGY SYSTEM

class BulkEnergySystem:
	'''Class responsible for balancing capacity between
	energy systems.'''
	def __init__(self, energy_systems: List[EnergySystem]):
		self.energy_systems = energy_systems

	def get_hourly_capacity(self):
		pass
