
from abc import ABC, abstractmethod

class Fleet(ABC):
	def __init__(self):
		self.nameplate_capacity = None
		self.forced_outage_rate = None

	@abstractmethod
	def dispatch(self):
		pass

class SampledFleet(Fleet):
	"""This is a class for fleets which are sample probabilistically.
	There should be parallel vectors for hourly capacity, and outage rates.
	"""
	pass

	def dispatch(self):
		pass

class DispatchedFleet(Fleet):
	"""This is a class for fleets which are not sampled probabilistically.
	For example, storage, demand response, etc."""
	pass

	def dispatch(self):
		pass
