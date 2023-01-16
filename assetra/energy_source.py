from __future__ import annotations
from abc import abstractmethod
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from energy_system import EnergySystem

log = getLogger(__name__)

class EnergySource:
	def __init__(self, energy_system: EnergySystem):
		self.energy_system = EnergySystem

	def assign_energy_system(self, energy_system: EnergySystem):
		self.energy_system = energy_system

	@property
	def hourly_generation(self):
		self.calculate_hourly_generation()
		return self._hourly_generation

	@abstractmethod
	def calculate_hourly_generation(self):
		pass

class StochasticEnergS(EnergySource):

	def calculate_hourly_generation(self):
		pass


