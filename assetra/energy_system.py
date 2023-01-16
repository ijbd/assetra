from logging import getLogger
from typing import List

from energy_source import EnergySource

log = getLogger(__name__)

class EnergySystem:
	def __init__(self):
		self.energy_sources = []

	def append_energy_source(self, energy_source:EnergySource):
		energy_source.assign_energy_system(self)
		self.energy_sources.append(energy_source)
		

	def pop_energy_source(self):
		return self.energy_sources.pop()

	def calculate_hourly_net_generation(self):
		for energy_source in self.energy_sources:
			energy_source.load_hourly_generation()
		
		