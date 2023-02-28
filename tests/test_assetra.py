import unittest
import sys

# external libraries
import xarray as xr

sys.path.append("..")

def get_sample_time_series(data):
    return xr.DataArray(
        data=[float(d) for d in data],
        coords=dict(
            time=xr.date_range(
                start="2016-01-01 00:00",
                periods=len(data),
                freq="H"
            )
        )
    )

def get_sample_net_capacity_matrix(data):
    return xr.DataArray(
        data=[[float(d) for d in row] for row in data],
        coords=dict(
            trial=[i for i in range(len(data))],
            time=xr.date_range(
                start="2016-01-01 00:00",
                periods=len(data[0]),
                freq="H"
            )
        )
    )

class TestCore(unittest.TestCase):

    def test_static_unit_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.core import StaticUnit

        # setup
        units = []
        units.append(StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 2])
        ))
        units.append(StaticUnit(
            id=2,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([3, 4])
        ))

        # test
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [1, 1]),
                hourly_capacity=(['energy_unit', 'time'], [[1, 2], [3, 4]])
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    periods=2,
                    freq="H"
                )
            )
        )
        observed = StaticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_static_unit_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.core import StaticUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [1, 1]),
                hourly_capacity=(['energy_unit', 'time'], [[1, 2], [3, 4]])
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    periods=2,
                    freq="H"
                )
            )
        )

        # test
        units = StaticUnit.from_unit_dataset(unit_dataset)
        expected = unit_dataset
        observed = StaticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_static_unit_probabilistic_capacity(self):
        """Static unit returns hourly capacity"""
        from assetra.core import StaticUnit

        # create unit
        unit = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 2])
        )
        unit_dataset = StaticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected=net_capacity_matrix.copy(data=[[1, 2]])
        observed=StaticUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.core import StochasticUnit

        units = []
        units.append(StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 2]),
            hourly_forced_outage_rate=get_sample_time_series([0.1, 0.2])
        ))
        units.append(StochasticUnit(
            id=2,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([3, 4]),
            hourly_forced_outage_rate=get_sample_time_series([0.3, 0.4])
        ))

        # test
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [1, 1]),
                hourly_capacity=(['energy_unit', 'time'], [[1, 2], [3, 4]]),
                hourly_forced_outage_rate=(['energy_unit', 'time'], [[0.1, 0.2], [0.3, 0.4]])
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    periods=2,
                    freq="H"
                )
            )
        )
        observed = StochasticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_stochastic_unit_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.core import StochasticUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [1, 1]),
                hourly_capacity=(['energy_unit', 'time'], [[1, 2], [3, 4]]),
                hourly_forced_outage_rate=(['energy_unit', 'time'], [[0.1, 0.2], [0.3, 0.4]])
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 00:00",
                    periods=2,
                    freq="H"
                )
            )
        )

        # test
        units = StochasticUnit.from_unit_dataset(unit_dataset)
        expected = unit_dataset
        observed = StochasticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))
       
    def test_stochastic_unit_probabilistic_capacity_for_0(self):
        """Stochastic unit with FOR of 0 has full capacity"""
        from assetra.core import StochasticUnit

        # create unit
        unit = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 1]),
            hourly_forced_outage_rate=get_sample_time_series([0, 0])
        )
        unit_dataset = StochasticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected=net_capacity_matrix.copy(data=[[1, 1]])
        observed=StochasticUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_probabilistic_capacity_for_1(self):
        """Stochastic unit with FOR of 1 has no capacity"""
        from assetra.core import StochasticUnit

        # create unit
        unit = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 1]),
            hourly_forced_outage_rate=get_sample_time_series([1, 1])
        )
        unit_dataset = StochasticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected=net_capacity_matrix.copy(data=[[0, 0]])
        observed=StochasticUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_probabilistic_capacity_for_tv(self):
        """Stochastic unit has time-varying FOR"""
        from assetra.core import StochasticUnit

        # create unit
        unit = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 1]),
            hourly_forced_outage_rate=get_sample_time_series([0, 1])
        )
        unit_dataset = StochasticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected=net_capacity_matrix.copy(data=[[1, 0]])
        observed=StochasticUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.core import StorageUnit

        units = []
        units.append(StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=2,
            charge_capacity=3,
            roundtrip_efficiency=0.8
        ))
        units.append(StorageUnit(
            id=2,
            nameplate_capacity=2,
            charge_rate=4,
            discharge_rate=5,
            charge_capacity=6,
            roundtrip_efficiency=0.9
        ))

        # test
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [1, 2]),
                charge_rate=(['energy_unit'], [1, 4]),
                discharge_rate=(['energy_unit'], [2, 5]),
                charge_capacity=(['energy_unit'], [3, 6]),
                roundtrip_efficiency=(['energy_unit'], [0.8, 0.9])
            ),
            coords=dict(
                energy_unit=[1, 2]
            )
        )
        observed = StorageUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_storage_unit_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.core import StorageUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [1, 2]),
                charge_rate=(['energy_unit'], [1, 4]),
                discharge_rate=(['energy_unit'], [2, 5]),
                charge_capacity=(['energy_unit'], [3, 6]),
                roundtrip_efficiency=(['energy_unit'], [0.8, 0.9])
            ),
            coords=dict(
                energy_unit=[1, 2]
            )
        )

        # test
        units = StorageUnit.from_unit_dataset(unit_dataset)
        expected = unit_dataset
        observed = StorageUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))
      
    def test_storage_unit_probabilistic_capacity_discharge_1(self):
        """Storage unit should not discharge more than its current capacity."""
        from assetra.core import StorageUnit

        # create unit
        unit = StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=1,
            charge_capacity=1,
            roundtrip_efficiency=1,
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, -1, -1, -1]])

        # test
        expected = net_capacity_matrix.copy(data=[[1, 0, 0, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_probabilistic_capacity_discharge_2(self):
        """Storage unit should not discharge more than its discharge rate."""
        from assetra.core import StorageUnit

        # create unit
        unit = StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=1,
            charge_capacity=3,
            roundtrip_efficiency=1,
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-2, -2, -2, -2]])

        # test
        expected = net_capacity_matrix.copy(data=[[1, 1, 1, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_probabilistic_capacity_charge(self):
        """Storage unit should charge as much as possible."""
        from assetra.core import StorageUnit

        # create unit
        unit = StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=1,
            charge_capacity=1,
            roundtrip_efficiency=1,
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, 1, 1, 1]])

        # test
        expected = net_capacity_matrix.copy(data=[[1, -1, 0, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_probabilistic_capacity_efficiency(self):
        """Storage unit is efficiency derated on charge and discharge"""
        from assetra.core import StorageUnit

        # create unit
        unit = StorageUnit(
            id=1,
            nameplate_capacity=4,
            charge_rate=1,
            discharge_rate=4,
            charge_capacity=4,
            roundtrip_efficiency=0.25
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, 1, 1, 1, 1, 1]])

        # test
        expected = net_capacity_matrix.copy(data=[[1, -1, -1, -1, -1, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(unit_dataset, net_capacity_matrix)
        self.assertTrue(expected.equals(observed))

    def test_system_builder_add_unit(self):
        """Energy units can be added and removed from systems."""
        from assetra.core import StaticUnit, EnergySystem

        e = EnergySystem()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0])
        )
        

        # sub-test 1
        self.assertEqual(e.size, 0)

        # sub-test 2
        e.add_unit(u)
        self.assertEqual(e.size, 1)

        # sub-test 3
        e.remove_unit(u)
        self.assertEqual(e.size, 0)

    def test_system_builder_duplicates(self):
        """Energy units should not be duplicated."""
        from assetra.core import StaticUnit, EnergySystem

        e = EnergySystem()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0])
        )

        # sub-test 1
        e.add_unit(u)
        self.assertRaises(RuntimeError, e.add_unit, u)
    
    def test_system_builder_build(self):
        pass

    def test_system_builder_from_system(self):
        pass

    def test_system_dataset_order(self):
        pass

    def test_system_save(self):
        pass

    def test_system_load(self):
        pass    
    
    #### OLD ######

    def test_ener_sys_old_3(self):
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

    def test_ener_sys_old_4(self):
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

    def test_ener_sys_old_5(self):
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
        u1 = StaticUnit(id=1, nameplate_capacity=1, hourly_capacity=[0, 0, -1, 0])
        e1.add_unit(u1)

        # additional system
        e2 = EnergySystem()
        u2 = StaticUnit(id=1, nameplate_capacity=1, hourly_capacity=[1, 1, 1, 1])
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
