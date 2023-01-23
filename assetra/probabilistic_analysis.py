from core import EnergySystem, BulkEnergySystem

class ProbabilisticSimulation:
	'''Class responsible for creating/storing the Monte Carlo
	trials for EnergySystem objects.'''
	def __init__(
		self, 
		energy_system: EnergySystem,
		num_monte_carlo_trials: int,
		num_time_steps: int):
		pass
		
	def run(self):
		pass

class BulkProbabilisticSimulation:
	'''Class responsible for creating/storing the Monte Carlo
	trials for BulkEnergySystem objects.'''
	def __init__(
		self,
		bulk_energy_system: BulkEnergySystem,
		num_monte_carlo_trials: int,
		num_time_steps: int):
		pass

	def run(self):
		pass
