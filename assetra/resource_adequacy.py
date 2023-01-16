from abc import abstractmethod
from logging import getLogger

from energy_system import EnergySystem

log = getLogger(__name__)

class ResourceAdequacy:
	def __init__(self, energy_system: EnergySystem):
		pass

	@abstractmethod
	def calculate_resource_adequacy(self):
		pass