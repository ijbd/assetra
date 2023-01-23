from abc import ABC

class ResourceContributionMetric(ABC):
	'''Class responsible for evaluating the resource contribution
	of one energy system object to another.'''
	def __init__(
		self):
		pass

	@abstractmethod
	def get_resource_contribution(self):
		pass
