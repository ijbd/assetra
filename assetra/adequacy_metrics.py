from abc import ABC

class ResourceAdequacyMetric(ABC):
	'''Class responsible for evaluating resource adequacy of
	a ProbabilisticSimulation object.'''
	def __init__(
		self):
		pass

	@abstractmethod
	def get_resource_adequacy(self):
		pass
