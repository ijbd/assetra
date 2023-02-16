import unittest
import sys
import logging
from pathlib import Path

# external libraries
import xarray as xr

sys.path.append("..")


class TestCore(unittest.TestCase):
    def test_demand_unit(self):
        """Capacity of demand unit is negation of demand."""
        from assetra.core import DemandUnit

        hourly_demand = xr.DataArray(
            data=[1, 2, 3],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = DemandUnit(id=1, hourly_demand=hourly_demand)

        # test
        expected = -hourly_demand
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 02:00"
        )
        self.assertTrue(expected.equals(observed))

    def test_constant_demand_unit(self):
        """Capacity of constant demand unit is constant."""
        from assetra.core import ConstantDemandUnit

        u = ConstantDemandUnit(id=1, demand=2)

        # test
        expected = xr.DataArray(
            data=[-2, -2, -2],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 02:00"
        )
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_1(self):
        """Capacity of stochastic unit with FOR=0 is full capacity."""
        from assetra.core import StochasticUnit

        hourly_capacity = xr.DataArray(
            data=[1, 1, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        hourly_forced_outage_rate = xr.DataArray(
            data=[0, 0, 0],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = hourly_capacity
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 02:00"
        )
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_2(self):
        """Capacity of stochastic unit with FOR=1 is zero."""
        from assetra.core import StochasticUnit

        hourly_capacity = xr.DataArray(
            data=[1, 1, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        hourly_forced_outage_rate = xr.DataArray(
            data=[1, 1, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = xr.zeros_like(hourly_capacity)
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 02:00"
        )
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_3(self):
        """FOR of stochastic unit is time-varying."""
        from assetra.core import StochasticUnit

        hourly_capacity = xr.DataArray(
            data=[1, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 01:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        hourly_forced_outage_rate = xr.DataArray(
            data=[1, 0],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 01:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=hourly_capacity,
            hourly_forced_outage_rate=hourly_forced_outage_rate,
        )

        # test
        expected = xr.DataArray(
            data=[0, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 01:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00"
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_1(self):
        """Storage unit should not overdischarge."""
        from assetra.core import StorageUnit

        net_hourly_capacity = xr.DataArray(
            data=[-1, -1, -1, -1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 03:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StorageUnit(
            id=1,
            charge_rate=1,
            discharge_rate=1,
            duration=1,
            roundtrip_efficiency=1,
        )

        # test
        expected = xr.DataArray(
            data=[1, 0, 0, 0],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 03:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 03:00",
            net_hourly_capacity=net_hourly_capacity,
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_2(self):
        """Storage unit should charge as much as possible."""
        from assetra.core import StorageUnit

        net_hourly_capacity = xr.DataArray(
            data=[-1, 1, 1, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 03:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StorageUnit(
            id=1,
            charge_rate=1,
            discharge_rate=1,
            duration=1,
            roundtrip_efficiency=1,
        )

        # test
        expected = xr.DataArray(
            data=[1, -1, 0, 0],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 03:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 03:00",
            net_hourly_capacity=net_hourly_capacity,
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_3(self):
        """Storage unit is efficiency-derated on charge and discharge."""
        from assetra.core import StorageUnit

        net_hourly_capacity = xr.DataArray(
            data=[-1, 1, 1, 1, 1, 1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 05:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StorageUnit(
            id=1,
            charge_rate=1,
            discharge_rate=4,
            duration=1,
            roundtrip_efficiency=0.25,
        )

        # test
        expected = xr.DataArray(
            data=[1, -1, -1, -1, -1, 0],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 05:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 05:00",
            net_hourly_capacity=net_hourly_capacity,
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_4(self):
        """Storage unit should not discharge more than its discharge rate."""
        from assetra.core import StorageUnit

        net_hourly_capacity = xr.DataArray(
            data=[-2, -2, -2, -2],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 03:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        u = StorageUnit(
            id=1,
            charge_rate=1,
            discharge_rate=1,
            duration=3,
            roundtrip_efficiency=1,
        )

        # test
        expected = xr.DataArray(
            data=[1, 1, 1, 0],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 03:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )
        observed = u.get_hourly_capacity(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 03:00",
            net_hourly_capacity=net_hourly_capacity,
        )
        self.assertTrue(expected.equals(observed))

    def test_energy_system_1(self):
        """Energy units can be added and removed from systems."""
        from assetra.core import StaticUnit, EnergySystem

        e = EnergySystem()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=xr.DataArray(
                data=[1, 2, 3],
                coords=dict(
                    time=xr.date_range(
                        start="2016-01-01 00:00",
                        end="2016-01-01 02:00",
                        freq="H",
                        inclusive="both",
                    )
                ),
            ),
        )

        # sub-test 1
        self.assertEqual(e.size, 0)

        # sub-test 2
        e.add_unit(u)
        self.assertEqual(e.size, 1)

        # sub-test 3
        e.remove_unit(u)
        self.assertEqual(e.size, 0)

    def test_energy_system_2(self):
        """Energy units should not be duplicated."""
        from assetra.core import StaticUnit, EnergySystem

        e = EnergySystem()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=xr.DataArray(
                data=[1, 2, 3],
                coords=dict(
                    time=xr.date_range(
                        start="2016-01-01 00:00",
                        end="2016-01-01 02:00",
                        freq="H",
                        inclusive="both",
                    )
                ),
            ),
        )

        # sub-test 1
        e.add_unit(u)
        self.assertRaises(RuntimeError, e.add_unit, u)

    def test_energy_system_3(self):
        """Energy units should be stable-sorted.
        i.e. storage units should be held in order of insert"""
        from assetra.core import StaticUnit, StorageUnit, EnergySystem

        e = EnergySystem()
        u1 = StorageUnit(
            id=1,
            charge_rate=1,
            discharge_rate=1,
            duration=3,
            roundtrip_efficiency=1,
        )
        u2 = StaticUnit(
            id=2,
            nameplate_capacity=1,
            hourly_capacity=xr.DataArray(
                data=[1, 2, 3],
                coords=dict(
                    time=xr.date_range(
                        start="2016-01-01 00:00",
                        end="2016-01-01 02:00",
                        freq="H",
                        inclusive="both",
                    )
                ),
            ),
        )
        u3 = StorageUnit(
            id=3,
            charge_rate=1,
            discharge_rate=1,
            duration=3,
            roundtrip_efficiency=1,
        )
        e.add_unit(u1)
        e.add_unit(u2)
        e.add_unit(u3)

        # sub-test 1
        expected = (u2, u1, u3)
        observed = e.energy_units
        self.assertTupleEqual(expected, observed)

    def test_energy_system_4(self):
        """Energy systems should return hourly capacity by unit."""
        from assetra.core import DemandUnit, StorageUnit, EnergySystem

        e = EnergySystem()
        e.add_unit(
            DemandUnit(
                id=1,
                hourly_demand=xr.DataArray(
                    data=[1, 1, 1],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 02:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )
        e.add_unit(
            StorageUnit(
                id=2,
                charge_rate=1,
                discharge_rate=1,
                duration=2,
                roundtrip_efficiency=1,
            )
        )
        e.add_unit(
            StorageUnit(
                id=3,
                charge_rate=1,
                discharge_rate=1,
                duration=1,
                roundtrip_efficiency=1,
            )
        )

        # test
        expected = xr.DataArray(
            data=[[-1, -1, -1], [1, 1, 0], [0, 0, 1]],
            coords=dict(
                energy_unit=[1, 2, 3],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                ),
            ),
        )
        observed = e.get_hourly_capacity_by_unit(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 02:00"
        )
        self.assertTrue(expected.equals(observed))

    def test_energy_system_5(self):
        """Energy systems should accept pre-existing net capacity."""
        from assetra.core import DemandUnit, StorageUnit, EnergySystem

        e = EnergySystem()
        e.add_unit(
            DemandUnit(
                id=1,
                hourly_demand=xr.DataArray(
                    data=[1, 1, 1],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 02:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )
        e.add_unit(
            StorageUnit(
                id=2,
                charge_rate=1,
                discharge_rate=1,
                duration=2,
                roundtrip_efficiency=1,
            )
        )
        e.add_unit(
            StorageUnit(
                id=3,
                charge_rate=1,
                discharge_rate=1,
                duration=1,
                roundtrip_efficiency=1,
            )
        )

        net_hourly_capacity = xr.DataArray(
            data=[1, 0, -1],
            coords=dict(
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                )
            ),
        )

        # sub-test 1
        expected = xr.DataArray(
            data=[[-1, -1, -1], [0, 1, 1], [0, 0, 1]],
            coords=dict(
                energy_unit=[1, 2, 3],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                ),
            ),
        )
        observed = e.get_hourly_capacity_by_unit(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 02:00",
            net_hourly_capacity=net_hourly_capacity,
        )
        self.assertTrue(expected.equals(observed))

        # sub-test 2
        expected = xr.zeros_like(net_hourly_capacity)
        observed = net_hourly_capacity
        self.assertTrue(expected.equals(observed))


class TestProbabilisticAnalysis(unittest.TestCase):
    def test_probabilistic_simulation_1(self):
        """Probabilistic simulation should generate hourly capacity matrix."""
        from assetra.core import EnergySystem, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        # create system
        e = EnergySystem()
        e.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=xr.DataArray(
                    data=[1, 1, 2],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 02:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )
        e.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=xr.DataArray(
                    data=[2, 2, 1],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 02:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )

        # create simulation
        ps = ProbabilisticSimulation(
            e, start_hour="2016-01-01 00:00", end_hour="2016-01-01 02:00", trial_size=3
        )

        # test
        ps.run()
        expected = xr.DataArray(
            data=[[[1, 1, 2], [2, 2, 1]]] * 3,
            coords=dict(
                trial=[0, 1, 2],
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                ),
            ),
        )
        observed = ps.hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

    def test_probabilistic_simulation_2(self):
        """Probabilistic simulation should allow flexible time-indexing."""
        from assetra.core import EnergySystem, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        # create system
        e = EnergySystem()
        e.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=xr.DataArray(
                    data=[1, 1, 2],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 02:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )
        e.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=xr.DataArray(
                    data=[2, 2, 1],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 02:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )

        # create simulation
        ps = ProbabilisticSimulation(
            e, start_hour="2016-01-01 01:00", end_hour="2016-01-01 02:00", trial_size=3
        )

        # test
        ps.run()
        expected = xr.DataArray(
            data=[[[1, 2], [2, 1]]] * 3,
            coords=dict(
                trial=[0, 1, 2],
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 01:00",
                    end="2016-01-01 02:00",
                    freq="H",
                    inclusive="both",
                ),
            ),
        )
        observed = ps.hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))


class TestResourceAdequacyMetric(unittest.TestCase):
    def test_eue_1(self):
        """Definition of EUE (single trial)"""
        from assetra.core import EnergySystem, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation
        from assetra.adequacy_metrics import ExpectedUnservedEnergy

        # create system
        e = EnergySystem()
        e.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=xr.DataArray(
                    data=[-1, -1],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 01:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )
        e.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=xr.DataArray(
                    data=[0, 1],
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00",
                            end="2016-01-01 01:00",
                            freq="H",
                            inclusive="both",
                        )
                    ),
                ),
            )
        )

        # create simulation
        ps = ProbabilisticSimulation(
            e, start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=3
        )
        ps.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # sub-test 1
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_2(self):
        """EUE should ignore excess capacity in non-loss-of-load hours"""
        from assetra.core import EnergySystem, DemandUnit, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation
        from assetra.adequacy_metrics import ExpectedUnservedEnergy

        # create system
        e = EnergySystem()
        u1 = DemandUnit(id=1, hourly_demand=np.array([1, 1]))
        u2 = StaticUnit(id=1, nameplate_capacity=1, hourly_capacity=np.array([0, 2]))
        e.add_unit(u1)
        e.add_unit(u2)

        # create simulation
        ps = ProbabilisticSimulation(e, start_hour=0, end_hour=2, trial_size=1)

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # sub-test 1
        expected = 0.5
        observed = ra.evaluate()
        self.assertEqual(expected, observed)


class TestResourceContribution(unittest.TestCase):
    def test_elcc_1(self):
        from assetra.core import StaticUnit, EnergySystem
        from assetra.probabilistic_analysis import ProbabilisticSimulation
        from assetra.adequacy_metrics import ExpectedUnservedEnergy
        from assetra.contribution_metrics import EffectiveLoadCarryingCapability

        # target system
        e1 = EnergySystem()
        u1 = StaticUnit(
            id=1, nameplate_capacity=1, hourly_capacity=np.array([0, 0, -1, 0])
        )
        e1.add_unit(u1)

        # additional system
        e2 = EnergySystem()
        u2 = StaticUnit(
            id=1, nameplate_capacity=1, hourly_capacity=np.array([1, 1, 1, 1])
        )
        e2.add_unit(u2)

        # simulation
        ps = ProbabilisticSimulation(e1, start_hour=0, end_hour=4, trial_size=1)
        ra = ExpectedUnservedEnergy(ps)
        rc = EffectiveLoadCarryingCapability(ra, e2, 0.01)

        # test
        expected = 1.0
        observed = rc.evaluate()
        self.assertEqual(expected, observed)


if __name__ == "__main__":
    unittest.main()
