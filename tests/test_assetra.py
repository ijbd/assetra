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
        u = DemandUnit(hourly_demand=hourly_demand)

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
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = np.array([0, 1])
        observed = u.get_hourly_capacity(start_hour=0, end_hour=2)
        self.assertTrue(np.array_equal(expected, observed))

    def test_storage_unit_1(self):
        """Storage unit should not overdischarge."""
        from assetra.core import StorageUnit

        net_hourly_capacity = np.array([-1, -1, -1, -1])
        u = StorageUnit(
            charge_rate=1,
            discharge_rate=1,
            duration=1,
            roundtrip_efficiency=1,
        )

        # test
        expected = np.array([1, 0, 0, 0])
        observed = u.get_hourly_capacity(net_hourly_capacity)
        self.assertTrue(np.array_equal(expected, observed))


    def test_storage_unit_2(self):
        """Storage unit should charge as much as possible."""
        from assetra.core import StorageUnit

        net_hourly_capacity = net_hourly_capacity = np.array([-1, 1, 1, 1])
        u = StorageUnit(
            charge_rate=1,
            discharge_rate=1,
            duration=1,
            roundtrip_efficiency=1,
        )

        # test
        expected = np.array([1, -1, 0, 0])
        observed = u.get_hourly_capacity(net_hourly_capacity)
        self.assertTrue(np.array_equal(expected, observed))

    def test_storage_unit_3(self):
        """Storage unit is efficiency-derated on charge and discharge."""
        from assetra.core import StorageUnit

        net_hourly_capacity = net_hourly_capacity = np.array([-1, 1, 1, 1, 1, 1])
        u = StorageUnit(
            charge_rate=1,
            discharge_rate=4,
            duration=1,
            roundtrip_efficiency=0.25,
        )

        # test
        expected = np.array([1, -1, -1, -1, -1, 0])
        observed = u.get_hourly_capacity(net_hourly_capacity)
        self.assertTrue(np.array_equal(expected, observed))

    def test_storage_unit_4(self):
        """Storage unit should not discharge more than its discharge rate."""
        from assetra.core import StorageUnit

        net_hourly_capacity = net_hourly_capacity = np.array([-2, -2, -2, -2])
        u = StorageUnit(
            charge_rate=1,
            discharge_rate=1,
            duration=3,
            roundtrip_efficiency=1,
        )

        # test
        expected = np.array([1, 1, 1, 0])
        observed = u.get_hourly_capacity(net_hourly_capacity)
        self.assertTrue(np.array_equal(expected, observed))

    def test_energy_system_1(self):
        """Energy units can be added and removed from systems."""
        from assetra.core import StaticUnit, EnergySystem

        e = EnergySystem()
        u = StaticUnit(nameplate_capacity=1, hourly_capacity=np.array([1]))

        # sub-test 1
        self.assertEqual(e.size, 0)

        # sub-test 2
        e.add_unit(u)
        self.assertEqual(e.size, 1)

        # sub-test 3
        e.remove_unit(u)
        self.assertEqual(e.size, 0)

    def test_energy_system_2(self):
        """Energy system returns generation matrix for energy units."""
        from assetra.core import StaticUnit, EnergySystem

        e = EnergySystem()
        u1 = StaticUnit(nameplate_capacity=1, hourly_capacity=np.array([0, 1]))
        u2 = StaticUnit(nameplate_capacity=4, hourly_capacity=np.array([3, 4]))

        # sub-test 1
        e.add_unit(u1)
        expected = np.array([[0, 1]])
        observed = e.get_hourly_capacity_by_unit(0, 2)
        self.assertTrue(np.array_equal(expected, observed))

        # sub-test 1
        e.add_unit(u2)
        expected = np.array([[0, 1], [3, 4]])
        observed = e.get_hourly_capacity_by_unit(0, 2)
        self.assertTrue(np.array_equal(expected, observed))


class TestProbabilisticAnalysis(unittest.TestCase):
    def test_probabilistic_simulation_1(self):
        """Probabilistic simulation should generate hourly capacity matrix."""
        from assetra.core import EnergySystem, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        # create system
        e = EnergySystem()
        u1 = StaticUnit(nameplate_capacity=1, hourly_capacity=np.array([0, 1]))
        u2 = StaticUnit(nameplate_capacity=4, hourly_capacity=np.array([3, 4]))
        e.add_unit(u1)
        e.add_unit(u2)

        # create simulation
        p = ProbabilisticSimulation(e, start_hour=0, end_hour=2, trial_size=4)

        # test
        p.run()
        expected = np.array([[[0, 1], [3, 4]]] * 4)
        observed = p.hourly_capacity_matrix
        self.assertTrue(np.array_equal(expected, observed))

    def test_probabilistic_simulation_2(self):
        """Removing units should not corrupt probabilistic simulation."""
        pass


class TestResourceAdequacyMetric(unittest.TestCase):
    def test_eue_1(self):
        """Definition of EUE (single trial)"""
        from assetra.core import EnergySystem, DemandUnit, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation
        from assetra.adequacy_metrics import ExpectedUnservedEnergy

        # create system
        e = EnergySystem()
        u1 = DemandUnit(hourly_demand=np.array([1, 1]))
        u2 = StaticUnit(nameplate_capacity=1, hourly_capacity=np.array([0, 1]))
        e.add_unit(u1)
        e.add_unit(u2)

        # create simulation
        p = ProbabilisticSimulation(e, start_hour=0, end_hour=2, trial_size=1)
        p.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(p)

        # sub-test 1
        expected = 0.5
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_2(self):
        """Definition of EUE (multiple trials)"""
        from assetra.core import EnergySystem, DemandUnit, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation
        from assetra.adequacy_metrics import ExpectedUnservedEnergy

        # create system
        e = EnergySystem()
        u1 = DemandUnit(hourly_demand=np.array([1, 1]))
        u2 = StaticUnit(nameplate_capacity=1, hourly_capacity=np.array([0, 1]))
        e.add_unit(u1)
        e.add_unit(u2)

        # create simulation
        p = ProbabilisticSimulation(e, start_hour=0, end_hour=2, trial_size=3)
        p.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(p)

        # sub-test 1
        expected = 0.5
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_3(self):
        """EUE should ignore excess capacity in non-loss-of-load hours"""
        from assetra.core import EnergySystem, DemandUnit, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation
        from assetra.adequacy_metrics import ExpectedUnservedEnergy

        # create system
        e = EnergySystem()
        u1 = DemandUnit(hourly_demand=np.array([1, 1]))
        u2 = StaticUnit(nameplate_capacity=1, hourly_capacity=np.array([0, 2]))
        e.add_unit(u1)
        e.add_unit(u2)

        # create simulation
        p = ProbabilisticSimulation(e, start_hour=0, end_hour=2, trial_size=1)
        p.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(p)

        # sub-test 1
        expected = 0.5
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

class TestResourceContribution(unittest.TestCase):
    def test_elcc_1(self):
        pass


if __name__ == "__main__":
    unittest.main()
