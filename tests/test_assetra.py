import unittest
import sys
from pathlib import Path

# external libraries
import numpy as np

sys.path.append("..")

class TestCore(unittest.TestCase):
    def test_demand_unit(self):
        """Capacity of demand unit is negation of demand."""
        from assetra.core import DemandUnit

        hourly_demand = np.array([1, 2, 3])
        u = DemandUnit(name="test_unit", hourly_demand=hourly_demand)

        # test
        expected = -hourly_demand
        observed = u.get_hourly_capacity(start_hour=0, end_hour=3)
        self.assertTrue(np.array_equal(expected, observed))

    def test_stochastic_unit_1(self):
        """Capacity of stochastic unit with FOR=0 is full capacity."""
        from assetra.core import StochasticUnit

        hourly_capacity = np.array([1, 1, 1])
        hourly_forced_outage_rate = np.array([0, 0, 0])
        u = StochasticUnit(
            name="test_unit",
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = hourly_capacity
        observed = u.get_hourly_capacity(start_hour=0, end_hour=3)
        self.assertTrue(np.array_equal(expected, observed))

    def test_stochastic_unit_2(self):
        """Capacity of stochastic unit with FOR=1 is zero."""
        from assetra.core import StochasticUnit

        hourly_capacity = np.array([1, 1, 1])
        hourly_forced_outage_rate = np.array([1, 1, 1])
        u = StochasticUnit(
            name="test_unit",
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = np.array([0, 0, 0])
        observed = u.get_hourly_capacity(start_hour=0, end_hour=3)
        self.assertTrue(np.array_equal(expected, observed))

    def test_stochastic_unit_3(self):
        """FOR of stochastic unit is time-varying."""
        from assetra.core import StochasticUnit

        hourly_capacity = np.array([1, 1])
        hourly_forced_outage_rate = np.array([1, 0])
        u = StochasticUnit(
            name="test_unit",
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = np.array([0, 1])
        observed = u.get_hourly_capacity(start_hour=0, end_hour=2)
        self.assertTrue(np.array_equal(expected, observed))

    def test_energy_system_1(self):
        """Energy units can be added and removed from systems."""
        from assetra.core import DemandUnit, EnergySystem

        e = EnergySystem(name="test_system")
        u1 = DemandUnit(name="test_unit", hourly_demand=np.array([0, 1]))

        # sub-test 1
        self.assertEqual(e.size, 0)

        # sub-test 2
        e.add_unit(u1)
        self.assertEqual(e.size, 1)

        # sub-test 3
        e.remove_unit(u1)
        self.assertEqual(e.size, 0)

    def test_energy_system_2(self):
        """Energy system returns generation matrix for energy units."""
        from assetra.core import DemandUnit, EnergySystem

        e = EnergySystem(name="test_system")
        u1 = DemandUnit(name="test_unit", hourly_demand=np.array([0, 1]))
        u2 = DemandUnit(name="test_unit", hourly_demand=np.array([3, 4]))

        # sub-test 1
        e.add_unit(u1)
        expected = np.array([[0, -1]])
        observed = e.get_hourly_capacity_by_unit(0, 2)
        self.assertTrue(np.array_equal(expected, observed))

        # sub-test 1
        e.add_unit(u2)
        expected = np.array([[0, -1], [-3, -4]])
        observed = e.get_hourly_capacity_by_unit(0, 2)
        self.assertTrue(np.array_equal(expected, observed))

class TestProbabilisticAnalysis(unittest.TestCase):
    def test_probabilistic_simulation_1(self):
        """Probabilistic simulation should generate hourly capacity matrix."""
        from assetra.core import EnergySystem, DemandUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        e = EnergySystem("test_system")
        u1 = DemandUnit(name="test_unit", hourly_demand=np.array([0, 1]))
        u2 = DemandUnit(name="test_unit", hourly_demand=np.array([3, 4]))
        e.add_unit(u1)
        e.add_unit(u2)

        p = ProbabilisticSimulation(e, 0, 2, 4)

        # test
        p.run()
        expected = np.array([[[0, -1], [-3, -4]]] * 4)
        observed = p.hourly_capacity_matrix
        self.assertTrue(np.array_equal(expected, observed))


if __name__ == "__main__":
    unittest.main()
