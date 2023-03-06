import pathlib
import unittest
import sys

# external libraries
import xarray as xr

sys.path.append("..")


def get_sample_time_series(data, start="2016-01-01 00:00"):
    return xr.DataArray(
        data=[float(d) for d in data],
        coords=dict(time=xr.date_range(start=start, periods=len(data), freq="H")),
    )


def get_sample_net_capacity_matrix(data, start="2016-01-01 00:00"):
    return xr.DataArray(
        data=[[float(d) for d in row] for row in data],
        coords=dict(
            trial=[i for i in range(len(data))],
            time=xr.date_range(start=start, periods=len(data[0]), freq="H"),
        ),
    )


class TestCore(unittest.TestCase):
    def test_static_unit_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.core import StaticUnit

        # setup
        units = []
        units.append(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 2]),
            )
        )
        units.append(
            StaticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([3, 4]),
            )
        )

        # test
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 1]),
                hourly_capacity=(["energy_unit", "time"], [[1, 2], [3, 4]]),
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(start="2016-01-01 00:00", periods=2, freq="H"),
            ),
        )
        observed = StaticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_static_unit_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.core import StaticUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 1]),
                hourly_capacity=(["energy_unit", "time"], [[1, 2], [3, 4]]),
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(start="2016-01-01 00:00", periods=2, freq="H"),
            ),
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
            id=1, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1, 2])
        )
        unit_dataset = StaticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected = get_sample_net_capacity_matrix([[1, 2]])
        observed = StaticUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.core import StochasticUnit

        units = []
        units.append(
            StochasticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 2]),
                hourly_forced_outage_rate=get_sample_time_series([0.1, 0.2]),
            )
        )
        units.append(
            StochasticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([3, 4]),
                hourly_forced_outage_rate=get_sample_time_series([0.3, 0.4]),
            )
        )

        # test
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 1]),
                hourly_capacity=(["energy_unit", "time"], [[1, 2], [3, 4]]),
                hourly_forced_outage_rate=(
                    ["energy_unit", "time"],
                    [[0.1, 0.2], [0.3, 0.4]],
                ),
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(start="2016-01-01 00:00", periods=2, freq="H"),
            ),
        )
        observed = StochasticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_stochastic_unit_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.core import StochasticUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 1]),
                hourly_capacity=(["energy_unit", "time"], [[1, 2], [3, 4]]),
                hourly_forced_outage_rate=(
                    ["energy_unit", "time"],
                    [[0.1, 0.2], [0.3, 0.4]],
                ),
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(start="2016-01-01 00:00", periods=2, freq="H"),
            ),
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
            hourly_forced_outage_rate=get_sample_time_series([0, 0]),
        )
        unit_dataset = StochasticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected = get_sample_net_capacity_matrix([[1, 1]])
        observed = StochasticUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_probabilistic_capacity_for_1(self):
        """Stochastic unit with FOR of 1 has no capacity"""
        from assetra.core import StochasticUnit

        # create unit
        unit = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 1]),
            hourly_forced_outage_rate=get_sample_time_series([1, 1]),
        )
        unit_dataset = StochasticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected = get_sample_net_capacity_matrix([[0, 0]])
        observed = StochasticUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_stochastic_unit_probabilistic_capacity_for_tv(self):
        """Stochastic unit has time-varying FOR"""
        from assetra.core import StochasticUnit

        # create unit
        unit = StochasticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 1]),
            hourly_forced_outage_rate=get_sample_time_series([0, 1]),
        )
        unit_dataset = StochasticUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected = get_sample_net_capacity_matrix([[1, 0]])
        observed = StochasticUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.core import StorageUnit

        units = []
        units.append(
            StorageUnit(
                id=1,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=2,
                charge_capacity=3,
                roundtrip_efficiency=0.8,
            )
        )
        units.append(
            StorageUnit(
                id=2,
                nameplate_capacity=2,
                charge_rate=4,
                discharge_rate=5,
                charge_capacity=6,
                roundtrip_efficiency=0.9,
            )
        )

        # test
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 2]),
                charge_rate=(["energy_unit"], [1, 4]),
                discharge_rate=(["energy_unit"], [2, 5]),
                charge_capacity=(["energy_unit"], [3, 6]),
                roundtrip_efficiency=(["energy_unit"], [0.8, 0.9]),
            ),
            coords=dict(energy_unit=[1, 2]),
        )
        observed = StorageUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_storage_unit_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.core import StorageUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 2]),
                charge_rate=(["energy_unit"], [1, 4]),
                discharge_rate=(["energy_unit"], [2, 5]),
                charge_capacity=(["energy_unit"], [3, 6]),
                roundtrip_efficiency=(["energy_unit"], [0.8, 0.9]),
            ),
            coords=dict(energy_unit=[1, 2]),
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
        expected = get_sample_net_capacity_matrix([[1, 0, 0, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
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
        expected = get_sample_net_capacity_matrix([[1, 1, 1, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
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
        expected = get_sample_net_capacity_matrix([[1, -1, 0, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
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
            roundtrip_efficiency=0.25,
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, 1, 1, 1, 1, 1]])

        # test
        expected = get_sample_net_capacity_matrix([[1, -1, -1, -1, -1, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_system_builder_add_unit(self):
        """Energy units can be added and removed from systems."""
        from assetra.core import StaticUnit, EnergySystemBuilder

        b = EnergySystemBuilder()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )

        # sub-test 1
        self.assertEqual(b.size, 0)

        # sub-test 2
        b.add_unit(u)
        self.assertEqual(b.size, 1)

        # sub-test 3
        b.remove_unit(u)
        self.assertEqual(b.size, 0)

    def test_system_builder_duplicates(self):
        """Energy units should not be duplicated."""
        from assetra.core import StaticUnit, EnergySystemBuilder

        b = EnergySystemBuilder()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )

        # test
        b.add_unit(u)
        self.assertRaises(RuntimeError, b.add_unit, u)

    def test_system_builder_valid_types(self):
        """System builder should only accept valid unit types."""
        from assetra.core import StaticUnit, EnergySystemBuilder

        class InvalidUnit(StaticUnit):
            pass

        b = EnergySystemBuilder()
        u = InvalidUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )

        # test
        self.assertRaises(RuntimeError, b.add_unit, u)

    def test_system_builder_build_single(self):
        """System builder should create all unit datasets."""
        from assetra.core import StaticUnit, EnergySystemBuilder

        b = EnergySystemBuilder()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )
        b.add_unit(u)

        e = b.build()

        # sub-test 1
        expected = [StaticUnit]
        observed = list(e.unit_datasets)
        self.assertEqual(expected, observed)

        # sub-test 2
        expected = e.unit_datasets[StaticUnit]
        observed = StaticUnit.to_unit_dataset([u])
        self.assertTrue(expected.equals(observed))

    def test_system_builder_build_full(self):
        """System should store unit datasets in order of dispatch."""
        from assetra.core import (
            StaticUnit,
            StochasticUnit,
            StorageUnit,
            EnergySystemBuilder,
        )

        b = EnergySystemBuilder()
        u1 = StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=1,
            charge_capacity=1,
            roundtrip_efficiency=0.8,
        )
        u2 = StochasticUnit(
            id=2,
            nameplate_capacity=2,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
            hourly_forced_outage_rate=get_sample_time_series([0.05, 0.05, 0.05]),
        )

        u3 = StaticUnit(
            id=3,
            nameplate_capacity=3,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )
        b.add_unit(u1)
        b.add_unit(u2)
        b.add_unit(u3)

        e = b.build()

        # sub-test 1
        expected = [StaticUnit, StochasticUnit, StorageUnit]
        observed = list(e.unit_datasets)
        self.assertEqual(expected, observed)

    def test_system_builder_from_system(self):
        """System builder should be recoverable from system"""
        from assetra.core import StaticUnit, EnergySystemBuilder

        b = EnergySystemBuilder()
        u1 = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 1, 1]),
        )
        b.add_unit(u1)
        e = b.build()
        b2 = EnergySystemBuilder.from_energy_system(e)

        # test
        expected = e.unit_datasets
        observed = b2.build().unit_datasets
        self.assertEqual(expected, observed)

    def test_system_save_load(self):
        """System should be recoverable from file system."""
        from assetra.core import (
            StaticUnit,
            StochasticUnit,
            EnergySystemBuilder,
            EnergySystem,
        )

        # setup directory
        save_dir = pathlib.Path("tmp-sys")
        save_dir.mkdir()
        try:
            b = EnergySystemBuilder()
            b.add_unit(
                StaticUnit(
                    id=1,
                    nameplate_capacity=1,
                    hourly_capacity=get_sample_time_series([1, 1, 1]),
                )
            )
            b.add_unit(
                StochasticUnit(
                    id=2,
                    nameplate_capacity=2,
                    hourly_capacity=get_sample_time_series([1, 1, 1]),
                    hourly_forced_outage_rate=get_sample_time_series([0.8, 0.8, 0.8]),
                )
            )
            b
            e = b.build()

            # save system
            e.save(save_dir)

            # load system
            e2 = EnergySystem()
            e2.load(save_dir)

            # test
            expected = e.unit_datasets
            observed = e2.unit_datasets
            self.assertEqual(expected, observed)
        except Exception as ex:
            raise ex
        finally:
            # delete artifacts
            for save_file in save_dir.iterdir():
                save_file.unlink()

            # delete dir
            save_dir.rmdir()

    def test_system_nameplate_capacity(self):
        """Energy system should return total nameplate capacity"""
        from assetra.core import StaticUnit, StochasticUnit, EnergySystemBuilder

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 1, 1]),
            )
        )
        b.add_unit(
            StochasticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 1, 1]),
                hourly_forced_outage_rate=get_sample_time_series([1, 1, 1]),
            )
        )
        e = b.build()

        # test
        expected = 2
        observed = e.nameplate_capacity
        self.assertEqual(expected, observed)

    def test_system_get_system_by_type(self):
        """Energy system should return total nameplate capacity"""
        from assetra.core import (
            EnergySystemBuilder,
            StaticUnit,
            StochasticUnit,
            StorageUnit,
            NONVOLATILE_UNIT_TYPES,
        )

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 1, 1]),
            )
        )
        b.add_unit(
            StochasticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 1, 1]),
                hourly_forced_outage_rate=get_sample_time_series([1, 1, 1]),
            )
        )
        b.add_unit(
            StorageUnit(
                id=3,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=1,
                roundtrip_efficiency=1,
            )
        )
        e = b.build()

        # sub-test 1
        expected = {
            StaticUnit: e.unit_datasets[StaticUnit],
            StochasticUnit: e.unit_datasets[StochasticUnit],
        }
        observed = e.get_system_by_type([StaticUnit, StochasticUnit]).unit_datasets
        self.assertEqual(expected, observed)

        # sub-test 2
        expected = {StorageUnit: e.unit_datasets[StorageUnit]}
        observed = e.get_system_by_type(StorageUnit).unit_datasets
        self.assertEqual(expected, observed)


class TestProbabilisticAnalysis(unittest.TestCase):
    def test_probabilistic_simulation_1(self):
        """Probabilistic simulation should generate hourly capacity matrix."""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 0, 1]),
            )
        )
        b.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=get_sample_time_series([2, 1, 1]),
            )
        )

        # build system
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 0:00", end_hour="2016-01-01 02:00", trial_size=3
        )
        ps.assign_energy_system(e)

        # test
        expected = get_sample_net_capacity_matrix([[3, 1, 2]] * 3)
        observed = ps.net_hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

    def test_probabilistic_simulation_2(self):
        """Probabilistic simulation should allow flexible time-indexing."""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 0, 1]),
            )
        )
        b.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=get_sample_time_series([2, 1, 1]),
            )
        )

        # build system
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 01:00", end_hour="2016-01-01 02:00", trial_size=4
        )
        ps.assign_energy_system(e)

        # test
        expected = get_sample_net_capacity_matrix(
            [[1, 2]] * 4, start="2016-01-01 01:00"
        )
        observed = ps.net_hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

    def test_probabilistic_simulation_3(self):
        """Probabilistic simulation should allow flexible time-indexing."""
        from assetra.core import EnergySystemBuilder, StaticUnit, StorageUnit
        from assetra.probabilistic_analysis import ProbabilisticSimulation

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-2, -2, -2]),
            )
        )
        b.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=get_sample_time_series([0, 2, 3]),
            )
        )
        b.add_unit(
            StorageUnit(
                id=3,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=1,
                roundtrip_efficiency=1,
            )
        )
        b.add_unit(
            StorageUnit(
                id=4,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=1,
                roundtrip_efficiency=1,
            )
        )

        # build system
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 0:00", end_hour="2016-01-01 02:00", trial_size=1
        )
        ps.assign_energy_system(e)

        # sub-test 1
        expected = get_sample_net_capacity_matrix(
            [[0, 0, 0]],
        )
        observed = ps.net_hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

        # sub-test 2
        expected = get_sample_net_capacity_matrix(
            [[-2, 0, 1]],
        )
        observed = ps.get_hourly_capacity_matrix_by_type(StaticUnit)

        # sub-test 2
        expected = get_sample_net_capacity_matrix(
            [[2, 0, -1]],
        )
        observed = ps.get_hourly_capacity_matrix_by_type(StorageUnit)


class TestResourceAdequacyMetric(unittest.TestCase):
    def test_eue_def(self):
        """Definition of EUE (single trial)"""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
        )

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1]),
            )
        )
        b.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=get_sample_time_series([0, 1]),
            )
        )
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=1
        )
        ps.assign_energy_system(e)

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_multi_trial(self):
        """Definition of EUE (multi-trial)"""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
        )

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1]),
            )
        )
        b.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=get_sample_time_series([0, 1]),
            )
        )
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=3
        )
        ps.assign_energy_system(e)

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_excess_capacity(self):
        """EUE should ignore excess capacity in non-loss-of-load hours"""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
        )

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1]),
            )
        )
        b.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=2,
                hourly_capacity=get_sample_time_series([0, 2]),
            )
        )
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=1
        )
        ps.assign_energy_system(e)

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_def(self):
        """Definition of LOLH (single trial)"""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            LossOfLoadHours,
        )

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, 0]),
            )
        )
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=1
        )
        ps.assign_energy_system(e)

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_multi_trial(self):
        """Definition of LOLH (multi trial)"""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            LossOfLoadHours,
        )

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, 0]),
            )
        )
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=3
        )
        ps.assign_energy_system(e)

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_magnitude(self):
        """LOLH does not account for magnitude of shortfall"""
        from assetra.core import EnergySystemBuilder, StaticUnit
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            LossOfLoadHours,
        )

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-3, 0]),
            )
        )
        e = b.build()

        # create simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00", end_hour="2016-01-01 01:00", trial_size=1
        )
        ps.assign_energy_system(e)

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)


class TestResourceContribution(unittest.TestCase):
    def test_elcc_ideal_generator(self):
        """ELCC of ideal generator is 1."""
        from assetra.core import StaticUnit, EnergySystemBuilder
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
            EffectiveLoadCarryingCapability,
        )

        # target system
        b1 = EnergySystemBuilder()
        b1.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([0, 0, -1, 0]),
            )
        )
        e1 = b1.build()

        # additional system
        b2 = EnergySystemBuilder()
        b2.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 1, 1, 1]),
            )
        )
        e2 = b2.build()

        # simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00:00",
            end_hour="2016-01-01 03:00:00",
            trial_size=1,
        )
        rc = EffectiveLoadCarryingCapability(e1, ps, ExpectedUnservedEnergy)

        # test
        expected = 1.0
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)

    def test_elcc_null_generator(self):
        """ELCC of zero capacity generator is 0."""
        from assetra.core import StaticUnit, EnergySystemBuilder
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
            EffectiveLoadCarryingCapability,
        )

        # target system
        b1 = EnergySystemBuilder()
        b1.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([0, 0, -1, 0]),
            )
        )
        e1 = b1.build()

        # additional system
        b2 = EnergySystemBuilder()
        b2.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([0, 0, 0, 0]),
            )
        )
        e2 = b2.build()

        # simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00:00",
            end_hour="2016-01-01 03:00:00",
            trial_size=1,
        )
        rc = EffectiveLoadCarryingCapability(e1, ps, ExpectedUnservedEnergy)

        # test
        expected = 0.0
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)

    def test_elcc_vol_addition(self):
        """ELCC should dispatch added volatile resources."""
        from assetra.core import StaticUnit, StorageUnit, EnergySystemBuilder
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
            EffectiveLoadCarryingCapability,
        )

        # target system
        b1 = EnergySystemBuilder()
        b1.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1, 1, 1]),
            )
        )
        e1 = b1.build()

        # additional system
        b2 = EnergySystemBuilder()
        b2.add_unit(
            StorageUnit(
                id=2,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=1,
                roundtrip_efficiency=1,
            )
        )
        e2 = b2.build()

        # simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00:00",
            end_hour="2016-01-01 03:00:00",
            trial_size=1,
        )
        rc = EffectiveLoadCarryingCapability(e1, ps, ExpectedUnservedEnergy)

        # test
        expected = 0.5
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)

    def test_elcc_vol_system(self):
        """ELCC should redispatch existing storage"""
        from assetra.core import StaticUnit, StorageUnit, EnergySystemBuilder
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
            EffectiveLoadCarryingCapability,
        )

        # target system
        b1 = EnergySystemBuilder()
        b1.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1]),
            )
        )
        b1.add_unit(
            StorageUnit(
                id=2,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=1,
                roundtrip_efficiency=1,
            )
        )
        e1 = b1.build()

        # additional system
        b2 = EnergySystemBuilder()
        b2.add_unit(
            StaticUnit(
                id=3,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([1, 0]),
            )
        )
        e2 = b2.build()

        # simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00:00",
            end_hour="2016-01-01 01:00:00",
            trial_size=1,
        )
        rc = EffectiveLoadCarryingCapability(e1, ps, ExpectedUnservedEnergy)

        # test
        expected = 0.5
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)

    def test_elcc_sequential(self):
        """ELCC should redispatch existing storage"""
        from assetra.core import StaticUnit, StorageUnit, EnergySystemBuilder
        from assetra.probabilistic_analysis import (
            ProbabilisticSimulation,
            ExpectedUnservedEnergy,
            EffectiveLoadCarryingCapability,
        )

        # target system
        b1 = EnergySystemBuilder()
        b1.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=0,
                hourly_capacity=get_sample_time_series([-1, 0, -1]),
            )
        )
        e1 = b1.build()

        # simulation
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00:00",
            end_hour="2016-01-01 02:00:00",
            trial_size=1,
        )
        rc = EffectiveLoadCarryingCapability(e1, ps, ExpectedUnservedEnergy)

        # additional system
        b2 = EnergySystemBuilder()
        b2.add_unit(
            StaticUnit(
                id=2,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([0, 1, 0]),
            )
        )
        e2 = b2.build()

        # sub-test 1
        expected = 0.0
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)

        # add storage unit
        b2.add_unit(
            StorageUnit(
                id=4,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=1,
                roundtrip_efficiency=1,
            )
        )
        e2 = b2.build()

        # sub-test 2
        expected = 0.666
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)

        # remove stochastic unit
        b2.remove_unit(b2.energy_units[0])
        e2 = b2.build()

        # sub-test 2
        expected = 0.333
        observed = rc.evaluate(e2)
        self.assertAlmostEqual(expected, observed, 2)


if __name__ == "__main__":
    unittest.main()
