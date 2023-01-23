import unittest 
import sys
from pathlib import Path

# external libraries
import numpy as np

sys.path.append('..')

class TestCore(unittest.TestCase):

	def test_demand_unit(self):
		'''The capacity of a demand unit is the negation of its 
		hourly demand'''
		from assetra.core import DemandUnit
		hourly_demand = np.array([1, 2, 3])
		u = DemandUnit(
			name='test_unit',
			hourly_demand=hourly_demand
			)

		# test
		expected = -hourly_demand
		observed = u.get_hourly_capacity()
		self.assertTrue(
			np.array_equal(
				expected,
				observed
				)
			)

	def test_stochastic_unit_1(self):
		'''The capacity of a stochastic unit with null forced outage
		rate is its full available capacity'''
		from assetra.core import StochasticUnit
		hourly_capacity = np.array([1, 1, 1])
		hourly_forced_outage_rate = np.array([0, 0, 0])
		u = StochasticUnit(
			name='test_unit',
			nameplate_capacity=1,
			hourly_capacity=hourly_capacity,
			hourly_forced_outage_rate=hourly_forced_outage_rate
			)

		# test
		expected = hourly_capacity
		observed = u.get_hourly_capacity()
		self.assertTrue(
			np.array_equal(
				expected,
				observed
				)
			)

	def test_stochastic_unit_2(self):
		'''The capacity of a stochastic unit with unity forced outage
		rate is zero'''
		from assetra.core import StochasticUnit
		hourly_capacity = np.array([1, 1, 1])
		hourly_forced_outage_rate = 1
		u = StochasticUnit(
			name='test_unit',
			nameplate_capacity=1,
			hourly_capacity=hourly_capacity,
			hourly_forced_outage_rate=hourly_forced_outage_rate
			)

		# test
		expected = np.array([0, 0, 0])
		observed = u.get_hourly_capacity()
		self.assertTrue(
			np.array_equal(
				expected,
				observed
				)
			)

if __name__ == '__main__':
	unittest.main()