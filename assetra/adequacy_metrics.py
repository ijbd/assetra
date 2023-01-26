from abc import abstractmethod, ABC

# package
from assetra.probabilistic_analysis import ProbabilisticSimulation

# external
import numpy as np

class ResourceAdequacyMetric(ABC):
	'''Class responsible for evaluating resource adequacy of
	a ProbabilisticSimulation object.'''
	def __init__(
		self,
		probabilistic_simulation: ProbabilisticSimulation):
		self._probabilistic_simulation = probabilistic_simulation

	@abstractmethod
	def evaluate(self):
		pass

class LossOfLoadDays(ResourceAdequacy):
	def evaluate(self):
		hourly_outages_by_trial = self.probabilisitic_simulation.hourly_outage_by_trial
		hourly_outages = np.sum(hourly_outages_by_trial, axis=0)
		hourly_outages_by_day = np.reshape(-1, 24)
		daily_outages = np.sum(hourly_outages_by_day, axis=1)
		return np.sum(daily_outages) / hourly_outages_by_trial.shape[1]

class LossOfLoadHours(ResourceAdequacyMetric):
	def evaluate(self):
		hourly_capacity_by_trial = \
			self.energy_system_prob_simulation.hourly_capacity_by_trial
		hourly_outages_by_trial = hourly_capacity_by_trial < 0
		return np.sum(hourly_outage_by_trial / hourly_outage_by_trial.shape[1])

class ExpectedUnservedEnergy(ResourceAdequacyMetric):
	def evaluate(self):
		net_hourly_capacity_by_trial = \
			self.energy_system_prob_simulation.net_hourly_capacity_by_trial
		hourly_unserved_energy = np.where(net_hourly_capacity_by_trial < 0,
			-net_hourly_capacity_by_trial,
			0)
		return np.sum(hourly_unserved_energy / hourly_outage_matrix.shape[1])

