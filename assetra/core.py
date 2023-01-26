from abc import abstractmethod, ABC
from logging import getLogger
from typing import List
from datetime import datetime

# external
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd

log = getLogger(__name__)

### ENERGY UNIT(S)

class EnergyUnit(ABC):
	def __init__(
		self,
		name: str,
		nameplate_capacity: float):
		self._name = str(name)
		self._nameplate_capacity = float(nameplate_capacity)

	# READ-ONLY VARIABLES

	@property
	def name(self):
		return self._name

	@property
	def nameplate_capacity(self):
		return self._nameplate_capacity 

	# METHODS

	@abstractmethod
	def get_hourly_capacity(self, start_hour: int, end_hour: int):
		'''Returns a single instance of the hourly capacity of the 
		generating unit.'''
		pass

class DemandUnit:
	'''Class responsible for returning capacity profile of fixed demand units
	(i.e. system loads).'''
	def __init__(self, name: str, hourly_demand: np.ndarray):
		EnergyUnit.__init__(self, name, nameplate_capacity=0)
		self._hourly_demand = hourly_demand

	def get_hourly_capacity(self, start_hour: int, end_hour: int):
		return -(self._hourly_demand[start_hour: end_hour])

class StochasticUnit:
	'''Class responsible for returning capacity profile of 
	stochastically-sampled units (i.e. generators).'''
	def __init__(self, name: str, nameplate_capacity: float, hourly_capacity: ArrayLike, 
		hourly_forced_outage_rate: ArrayLike):
		# initialize base class variables
		EnergyUnit.__init__(self, name, nameplate_capacity)
		# initialize stochastic specific variables
		self._hourly_capacity = hourly_capacity
		self._hourly_forced_outage_rate = hourly_forced_outage_rate

	def get_hourly_capacity(self, start_hour: int, end_hour: int):			
		hourly_outage_samples = np.random.random_sample(
			end_hour - start_hour)
		hourly_capacity_instance = np.where(
			hourly_outage_samples > self._hourly_forced_outage_rate[
				start_hour:end_hour],
			self._hourly_capacity[start_hour:end_hour],
			0)
		return hourly_capacity_instance

### ENERGY SYSTEM

class EnergySystem:
	'''Class responsible for managing energy units.'''
	def __init__(
		self,
		name: str):
		self._energy_units = []

	@property
	def size(self):
		return len(self._energy_units)

	@property
	def capacity(self):
		return sum([u.nameplate_capacity for u in self._energy_units])

	def add_unit(self, energy_unit: EnergyUnit):
		self._energy_units.append(energy_unit)

	def remove_unit(self, energy_unit: EnergyUnit):
		self._energy_units.remove(energy_unit)

	def get_hourly_capacity_by_unit(self, start_hour: int, end_hour: int):
		'''Returns the hourly capacity of each generating unit
		in the energy system.'''
		hourly_capacity_matrix = np.zeros(
			(self.size, end_hour - start_hour))
		hourly_net_capacity = np.zeros(end_hour - start_hour)
		for i, energy_unit in enumerate(self._energy_units):
			hourly_capacity_matrix[i] = energy_unit.get_hourly_capacity(start_hour, end_hour)
			hourly_net_capacity += hourly_capacity_matrix[i]

		return hourly_capacity_matrix

### BULK ENERGY SYSTEM

class BulkEnergySystem:
	'''Class responsible for balancing capacity between
	energy systems.'''
	def __init__(self, energy_systems: List[EnergySystem]):
		self.energy_systems = energy_systems

	def get_hourly_capacity(self, start_hour: int, end_hour: int):
		pass
