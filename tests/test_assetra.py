import pathlib
import unittest
import sys
import logging

# external libraries
import xarray as xr

sys.path.append("..")

logging.basicConfig()


def get_sample_time_series(data, start="2016-01-01 00:00"):
    return xr.DataArray(
        data=[float(d) for d in data],
        coords=dict(
            time=xr.date_range(start=start, periods=len(data), freq="h")
        ),
    )


def get_sample_net_capacity_matrix(data, start="2016-01-01 00:00"):
    return xr.DataArray(
        data=[[float(d) for d in row] for row in data],
        coords=dict(
            trial=[i for i in range(len(data))],
            time=xr.date_range(start=start, periods=len(data[0]), freq="h"),
        ),
    )


class TestAssetraUnits(unittest.TestCase):
    def test_static_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.units import StaticUnit

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
                time=xr.date_range(
                    start="2016-01-01 00:00", periods=2, freq="h"
                ),
            ),
        )
        observed = StaticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_static_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.units import StaticUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 1]),
                hourly_capacity=(["energy_unit", "time"], [[1, 2], [3, 4]]),
            ),
            coords=dict(
                energy_unit=[1, 2],
                time=xr.date_range(
                    start="2016-01-01 00:00", periods=2, freq="h"
                ),
            ),
        )

        # test
        units = StaticUnit.from_unit_dataset(unit_dataset)
        expected = unit_dataset
        observed = StaticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_static_probabilistic_capacity(self):
        """Static unit returns hourly capacity"""
        from assetra.units import StaticUnit

        # create unit
        unit = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([1, 2]),
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

    def test_demand_probabilistic_capacity(self):
        """Demand unit returns negative hourly demand"""
        from assetra.units import DemandUnit

        # create unit
        unit = DemandUnit(
            id=1,
            hourly_demand=get_sample_time_series([1, 2]),
        )
        unit_dataset = DemandUnit.to_unit_dataset([unit])

        # get net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[0, 0]])

        # test
        expected = get_sample_net_capacity_matrix([[-1, -2]])
        observed = DemandUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_stoch_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.units import StochasticUnit

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
                time=xr.date_range(
                    start="2016-01-01 00:00", periods=2, freq="h"
                ),
            ),
        )
        observed = StochasticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_stoch_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.units import StochasticUnit

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
                time=xr.date_range(
                    start="2016-01-01 00:00", periods=2, freq="h"
                ),
            ),
        )

        # test
        units = StochasticUnit.from_unit_dataset(unit_dataset)
        expected = unit_dataset
        observed = StochasticUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_stoch_probabilistic_capacity_for_0(self):
        """Stochastic unit with FOR of 0 has full capacity"""
        from assetra.units import StochasticUnit

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

    def test_stoch_probabilistic_capacity_for_1(self):
        """Stochastic unit with FOR of 1 has no capacity"""
        from assetra.units import StochasticUnit

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

    def test_stoch_probabilistic_capacity_for_tv(self):
        """Stochastic unit has time-varying FOR"""
        from assetra.units import StochasticUnit

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

    def test_storage_list_to_dataset(self):
        """Generate xarray dataset from unit list"""
        from assetra.units import StorageUnit

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
        # initial_soc is unspecified above, so it defaults to 1.0
        expected = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 2]),
                charge_rate=(["energy_unit"], [1, 4]),
                discharge_rate=(["energy_unit"], [2, 5]),
                charge_capacity=(["energy_unit"], [3, 6]),
                roundtrip_efficiency=(["energy_unit"], [0.8, 0.9]),
                initial_soc=(["energy_unit"], [1.0, 1.0]),
            ),
            coords=dict(energy_unit=[1, 2]),
        )
        observed = StorageUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_storage_dataset_to_list(self):
        """Generate unit list from xarray dataset"""
        from assetra.units import StorageUnit

        # setup
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(["energy_unit"], [1, 2]),
                charge_rate=(["energy_unit"], [1, 4]),
                discharge_rate=(["energy_unit"], [2, 5]),
                charge_capacity=(["energy_unit"], [3, 6]),
                roundtrip_efficiency=(["energy_unit"], [0.8, 0.9]),
                initial_soc=(["energy_unit"], [1.0, 1.0]),
            ),
            coords=dict(energy_unit=[1, 2]),
        )

        # test
        units = StorageUnit.from_unit_dataset(unit_dataset)
        expected = unit_dataset
        observed = StorageUnit.to_unit_dataset(units)
        self.assertTrue(observed.equals(expected))

    def test_storage_dataset_to_list_without_initial_soc(self):
        """Datasets saved before initial_soc existed should default to 1.0"""
        from assetra.units import StorageUnit

        # setup, omitting initial_soc as legacy saved systems do
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
        self.assertEqual([unit.initial_soc for unit in units], [1.0, 1.0])

    def test_storage_probabilistic_capacity_over_discharge(self):
        """Storage unit should not discharge more than its current capacity."""
        from assetra.units import StorageUnit

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

    def test_storage_probabilistic_capacity_discharge_rate(self):
        """Storage unit should not discharge more than its discharge rate."""
        from assetra.units import StorageUnit

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
        from assetra.units import StorageUnit

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
        from assetra.units import StorageUnit

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
        net_capacity_matrix = get_sample_net_capacity_matrix(
            [[-1, 1, 1, 1, 1, 1]]
        )

        # test
        expected = get_sample_net_capacity_matrix([[1, -1, -1, -1, -1, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_unit_multi_cycle(self):
        """Storage unit should operate more than one charge/discharge cycle."""
        from assetra.units import StorageUnit

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
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, 1, -1, 1]])

        # test
        expected = get_sample_net_capacity_matrix([[1, -1, 1, -1]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_partial_initial_soc(self):
        """Storage unit discharges only the energy given by its initial SOC."""
        from assetra.units import StorageUnit

        # create unit holding 2 units of energy (50% of 4) at hour zero
        unit = StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=1,
            charge_capacity=4,
            roundtrip_efficiency=1,
            initial_soc=0.5,
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, -1, -1, -1]])

        # test, discharging 1 unit of power for 2 hours before depleting
        expected = get_sample_net_capacity_matrix([[1, 1, 0, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_empty_initial_soc(self):
        """Storage unit starting empty cannot discharge until it charges."""
        from assetra.units import StorageUnit

        # create unit
        unit = StorageUnit(
            id=1,
            nameplate_capacity=1,
            charge_rate=1,
            discharge_rate=1,
            charge_capacity=1,
            roundtrip_efficiency=1,
            initial_soc=0.0,
        )
        unit_dataset = StorageUnit.to_unit_dataset([unit])

        # create net capacity matrix
        net_capacity_matrix = get_sample_net_capacity_matrix([[-1, 1, -1, -1]])

        # test, no discharge in hour zero since the unit starts empty
        expected = get_sample_net_capacity_matrix([[0, -1, 1, 0]])
        observed = StorageUnit.get_probabilistic_capacity_matrix(
            unit_dataset, net_capacity_matrix
        )
        self.assertTrue(expected.equals(observed))

    def test_storage_invalid_initial_soc(self):
        """Initial SOC outside the range 0 to 1 should be rejected."""
        from assetra.units import StorageUnit

        for invalid_soc in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                StorageUnit(
                    id=1,
                    nameplate_capacity=1,
                    charge_rate=1,
                    discharge_rate=1,
                    charge_capacity=1,
                    roundtrip_efficiency=1,
                    initial_soc=invalid_soc,
                )


class TestAssetraSystem(unittest.TestCase):
    def test_system_builder_add_unit(self):
        """Energy units can be added and removed from systems."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder

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
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder

        b = EnergySystemBuilder()
        u = StaticUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )

        # test
        b.add_unit(u)
        self.assertRaises(RuntimeWarning, b.add_unit, u)

    def test_system_builder_invalid_type(self):
        """System builder should only accept valid unit types."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder

        class InvalidUnit(StaticUnit):
            pass

        b = EnergySystemBuilder()
        u = InvalidUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )

        # test
        self.assertRaises(RuntimeWarning, b.add_unit, u)

    def test_system_builder_valid_type(self):
        """System builder should only accept valid unit types."""
        from assetra.units import StaticUnit, RESPONSIVE_UNIT_TYPES
        from assetra.system import EnergySystemBuilder

        class ValidUnit(StaticUnit):
            pass

        RESPONSIVE_UNIT_TYPES.append(ValidUnit)

        b = EnergySystemBuilder()
        u = ValidUnit(
            id=1,
            nameplate_capacity=1,
            hourly_capacity=get_sample_time_series([0, 0, 0]),
        )

        # test no raised error
        b.add_unit(u)

    def test_system_builder_build_single(self):
        """System builder should create all unit datasets."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder

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
        from assetra.units import StaticUnit, StochasticUnit, StorageUnit
        from assetra.system import EnergySystemBuilder

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
            hourly_forced_outage_rate=get_sample_time_series(
                [0.05, 0.05, 0.05]
            ),
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
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder

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
        from assetra.units import StaticUnit, StochasticUnit
        from assetra.system import EnergySystemBuilder, EnergySystem

        # setup directory
        save_dir = pathlib.Path("tmp-sys")
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
                    hourly_forced_outage_rate=get_sample_time_series(
                        [0.8, 0.8, 0.8]
                    ),
                )
            )
            
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

    def test_system_capacity(self):
        """Energy system should return total system capacity"""
        from assetra.units import StaticUnit, StochasticUnit
        from assetra.system import EnergySystemBuilder

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
        observed = e.system_capacity
        self.assertEqual(expected, observed)

    def test_system_get_system_by_type(self):
        """Energy system should return total nameplate capacity"""
        from assetra.units import StaticUnit, StochasticUnit, StorageUnit
        from assetra.system import EnergySystemBuilder

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
        observed = e.get_system_by_type(
            [StaticUnit, StochasticUnit]
        ).unit_datasets
        self.assertEqual(expected, observed)

        # sub-test 2
        expected = {StorageUnit: e.unit_datasets[StorageUnit]}
        observed = e.get_system_by_type(StorageUnit).unit_datasets
        self.assertEqual(expected, observed)


class TestAssetraSimulation(unittest.TestCase):
    def test_capacity_matrix(self):
        """Probabilistic simulation should generate hourly capacity matrix."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation

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
            start_hour="2016-01-01 0:00",
            end_hour="2016-01-01 02:00",
            trial_size=3,
        )
        ps.assign_energy_system(e)
        ps.run()

        # test
        expected = get_sample_net_capacity_matrix([[3, 1, 2]] * 3)
        observed = ps.net_hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

    def test_time_indexing(self):
        """Probabilistic simulation should allow flexible time-indexing."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation

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
            start_hour="2016-01-01 01:00",
            end_hour="2016-01-01 02:00",
            trial_size=4,
        )
        ps.assign_energy_system(e)
        ps.run()

        # test
        expected = get_sample_net_capacity_matrix(
            [[1, 2]] * 4, start="2016-01-01 01:00"
        )
        observed = ps.net_hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

    def test_by_type(self):
        """Probabilistic simulation should allow flexible time-indexing."""
        from assetra.units import StaticUnit, StorageUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation

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
            start_hour="2016-01-01 0:00",
            end_hour="2016-01-01 02:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

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

    def test_custom_type(self):
        """Probabilistic simulation should recognize added types."""
        from assetra.units import StaticUnit, StorageUnit, RESPONSIVE_UNIT_TYPES
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation

        # create system
        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1, -1]),
            )
        )

        # create custom type
        class CustomStorageUnit(StorageUnit):
            pass

        RESPONSIVE_UNIT_TYPES.append(CustomStorageUnit)

        b.add_unit(
            CustomStorageUnit(
                id=2,
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
            start_hour="2016-01-01 0:00",
            end_hour="2016-01-01 02:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # sub-test 1
        expected = get_sample_net_capacity_matrix([[0, -1, -1]])
        observed = ps.net_hourly_capacity_matrix
        self.assertTrue(expected.equals(observed))

        # sub-test 2
        expected = get_sample_net_capacity_matrix([[1, 0, 0]])
        observed = ps.get_hourly_capacity_matrix_by_type(CustomStorageUnit)


class TestAssetraMetrics(unittest.TestCase):
    def test_eue_def(self):
        """Definition of EUE (single trial)"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy

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
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_multi_trial(self):
        """Definition of EUE (multi-trial)"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy

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
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=3,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_eue_excess_capacity(self):
        """EUE should ignore excess capacity in non-loss-of-load hours"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy

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
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = ExpectedUnservedEnergy(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_def(self):
        """Definition of LOLH (single trial)"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadHours

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
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_multi_trial(self):
        """Definition of LOLH (multi trial)"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadHours

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
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=3,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_magnitude(self):
        """LOLH does not account for magnitude of shortfall"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadHours

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
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolh_fractional(self):
        """Definition of LOLH (fraction)"""
        from assetra.system import EnergySystem
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadHours

        e = EnergySystem()
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=2,
        )
        ps.assign_energy_system(e)
        ps.run()

        ps.run(get_sample_net_capacity_matrix([[-1, 0], [0, 0]]))

        # create adequacy model
        ra = LossOfLoadHours(ps)

        # test
        expected = 0.5
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lold_single_event(self):
        """LOLD should be 1 for a single event in one day"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadDays

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1] + [0] * 23),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 23:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadDays(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lold_single_event(self):
        """LOLD does not characterize magnitude of events"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadDays

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-2] + [0] * 23),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 23:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadDays(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lold_single_day_multi_event(self):
        """LOLD should be 1 for multiple events in one day"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadDays

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1] * 2 + [0] * 22),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 23:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadDays(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lold_multi_day_multi_event(self):
        """LOLD should be 2 for multiple events in two days"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadDays

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series(
                    [0] * 22 + [-1] * 4 + [0] * 22
                ),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-02 23:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadDays(ps)

        # test
        expected = 2
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lold_fractional(self):
        """Definition of LOLD (fraction)"""
        from assetra.system import EnergySystem
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadDays

        e = EnergySystem()
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-03 23:00",
            trial_size=2,
        )
        ps.assign_energy_system(e)
        ps.run()

        net_hourly_capacity_matrix = get_sample_net_capacity_matrix(
            [[0] * 24 * 3] * 2
        )
        # first trial lold = 2
        net_hourly_capacity_matrix[0, 23:25] = -1
        # second trial lold = 1
        net_hourly_capacity_matrix[1, 24 * 2] = -1
        ps.run(net_hourly_capacity_matrix)

        # create adequacy model
        ra = LossOfLoadDays(ps)

        # test
        expected = 1.5
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolf_single_hour_single_event(self):
        """Definition of LOLF"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadFrequency

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, 0]),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadFrequency(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolf_magnitude(self):
        """Definition of LOLF"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadFrequency

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-2, 0]),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadFrequency(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolf_single_hour_multi_event(self):
        """LOLF captures multiple non-contiguous single-hour events"""
        """Definition of LOLF"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadFrequency

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, 0, -1]),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 02:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadFrequency(ps)

        # test
        expected = 2
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolf_multi_hour_single_event(self):
        """LOLF captures non-contiguous multi-hour events"""
        """Definition of LOLF"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadFrequency

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, -1]),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 01:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadFrequency(ps)

        # test
        expected = 1
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolf_multi_hour_multi_event(self):
        """LOLF captures multiple non-contiguous multi-hour events"""
        """Definition of LOLF"""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadFrequency

        b = EnergySystemBuilder()
        b.add_unit(
            StaticUnit(
                id=1,
                nameplate_capacity=1,
                hourly_capacity=get_sample_time_series([-1, 0, -1, -1]),
            )
        )
        e = b.build()

        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 03:00",
            trial_size=1,
        )
        ps.assign_energy_system(e)
        ps.run()

        # create adequacy model
        ra = LossOfLoadFrequency(ps)

        # test
        expected = 2
        observed = ra.evaluate()
        self.assertEqual(expected, observed)

    def test_lolf_fractional(self):
        """LOLF is averaged across trials"""
        from assetra.system import EnergySystem
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import LossOfLoadFrequency

        e = EnergySystem()
        ps = ProbabilisticSimulation(
            start_hour="2016-01-01 00:00",
            end_hour="2016-01-01 4:00",
            trial_size=2,
        )
        ps.assign_energy_system(e)
        ps.run()
        ps.run(
            get_sample_net_capacity_matrix(
                [[-1, 0, -1, 0, -1], [0, -1, -1, -1, 0]]
            )
        )

        # create adequacy model
        ra = LossOfLoadFrequency(ps)

        # test
        expected = 2
        observed = ra.evaluate()
        self.assertEqual(expected, observed)


class TestAssetraContribution(unittest.TestCase):
    def test_elcc_ideal_generator(self):
        """ELCC of ideal generator is 1."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import EffectiveLoadCarryingCapability

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
        self.assertAlmostEqual(expected, observed, delta=0.01)

    def test_elcc_null_generator(self):
        """ELCC of zero capacity generator is 0."""
        from assetra.units import StaticUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import EffectiveLoadCarryingCapability

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
        self.assertAlmostEqual(expected, observed, delta=0.01)

    def test_elcc_resp_addition(self):
        """ELCC should dispatch added responsive resources."""
        from assetra.units import StaticUnit, StorageUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import EffectiveLoadCarryingCapability

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
        self.assertAlmostEqual(expected, observed, delta=0.01)

    def test_elcc_resp_system(self):
        """ELCC should redispatch original responsive fleet."""
        from assetra.units import StaticUnit, StorageUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import EffectiveLoadCarryingCapability

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
        self.assertAlmostEqual(expected, observed, delta=0.01)

    def test_elcc_sequential(self):
        """ELCC should redispatch existing storage"""
        from assetra.units import StaticUnit, StorageUnit
        from assetra.system import EnergySystemBuilder
        from assetra.simulation import ProbabilisticSimulation
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import EffectiveLoadCarryingCapability

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
        self.assertAlmostEqual(expected, observed, delta=0.01)


class TestAssetraMarginalContribution(unittest.TestCase):
    def _system(self, *units):
        """Return an energy system built from the given units."""
        from assetra.system import EnergySystemBuilder

        builder = EnergySystemBuilder()
        for unit in units:
            builder.add_unit(unit)
        return builder.build()

    def _simulation(self, hours: int, trials: int):
        """Return a simulation of the first hours of 2016-01-01."""
        from assetra.simulation import ProbabilisticSimulation

        return ProbabilisticSimulation(
            start_hour="2016-01-01 00:00:00",
            end_hour=f"2016-01-01 {hours - 1:02d}:00:00",
            trial_size=trials,
        )

    def _f1(self, extra_hours=0, extra_units=(), demand=(110, 120, 80, 70)):
        """Linear regime: 4 hours, 2 trials; the resources are Static 2 and the
        whole static fleet.

        extra_hours appends that many trailing hours of 0 to every profile,
        while the simulation stays at 4 hours. extra_units are added to the
        base after Static 2.
        """
        from assetra.units import DemandUnit, StaticUnit

        tail = [0] * extra_hours
        static2 = StaticUnit(id=2, nameplate_capacity=10, hourly_capacity=get_sample_time_series([5, 10, 0, 10] + tail))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series(list(demand) + tail)),
            StaticUnit(id=1, nameplate_capacity=100, hourly_capacity=get_sample_time_series([100] * 4 + tail)),
            static2,
            *extra_units,
        )
        resources = {"static2": self._system(static2), "fleet": e1.get_system_by_type(StaticUnit)}
        return e1, self._simulation(4, 2), resources

    def _f2(self):
        """Kink at net = 0: 3 hours, 1 trial, Static 2 is the resource."""
        from assetra.units import DemandUnit, StaticUnit

        static2 = StaticUnit(id=2, nameplate_capacity=2, hourly_capacity=get_sample_time_series([1, 0, 2]))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([15, 10, 6])),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10] * 3)),
            static2,
        )
        return e1, self._simulation(3, 1), {"static2": self._system(static2)}

    def _f3(self):
        """Stochastic units with FOR 0 or 1: 4 hours, 3 trials; the resources are g (id 2), h (id 3) and both."""
        from assetra.units import DemandUnit, StaticUnit, StochasticUnit

        g = StochasticUnit(id=2, nameplate_capacity=4, hourly_capacity=get_sample_time_series([4] * 4),
                           hourly_forced_outage_rate=get_sample_time_series([0, 1, 1, 1]))
        h = StochasticUnit(id=3, nameplate_capacity=2, hourly_capacity=get_sample_time_series([2] * 4),
                           hourly_forced_outage_rate=get_sample_time_series([0, 0, 1, 1]))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([57, 62, 30, 10])),
            StaticUnit(id=1, nameplate_capacity=50, hourly_capacity=get_sample_time_series([50] * 4)),
            g,
            h,
        )
        resources = {"g": self._system(g), "h": self._system(h), "gh": e1.get_system_by_type(StochasticUnit)}
        return e1, self._simulation(4, 3), resources

    def _f4(self):
        """The resource's own random outages: 4 hours, 10 trials, Stochastic 2 is the resource."""
        from assetra.units import DemandUnit, StaticUnit, StochasticUnit

        stoch2 = StochasticUnit(id=2, nameplate_capacity=4, hourly_capacity=get_sample_time_series([4] * 4),
                                hourly_forced_outage_rate=get_sample_time_series([0.5] * 4))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([12] * 4)),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10] * 4)),
            stoch2,
        )
        return e1, self._simulation(4, 10), {"stoch2": self._system(stoch2)}

    def _f5(self):
        """Storage: 2 hours, 1 trial; the resources are Storage 2, Static 1 and both together."""
        from assetra.units import DemandUnit, StaticUnit, StorageUnit

        static1 = StaticUnit(id=1, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1, 0]))
        storage2 = StorageUnit(id=2, nameplate_capacity=1, charge_rate=1, discharge_rate=1, charge_capacity=1,
                               roundtrip_efficiency=1.0, initial_soc=1)
        e1 = self._system(DemandUnit(id=0, hourly_demand=get_sample_time_series([2, 1])), static1, storage2)
        resources = {"storage2": self._system(storage2), "static1": self._system(static1),
                     "hybrid": self._system(static1, storage2)}
        return e1, self._simulation(2, 1), resources

    def _f6(self):
        """Storage factor above 1: 2 hours, 1 trial, Storage 2 is the resource."""
        from assetra.units import DemandUnit, StaticUnit, StorageUnit

        storage2 = StorageUnit(id=2, nameplate_capacity=1, charge_rate=2, discharge_rate=2, charge_capacity=6,
                               roundtrip_efficiency=1.0, initial_soc=1)
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([4, 4])),
            StaticUnit(id=1, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1, 1])),
            storage2,
        )
        return e1, self._simulation(2, 1), {"storage2": self._system(storage2)}

    def _f7(self):
        """Hydro: 3 hours in January, 1 trial, Hydro 2 is the resource."""
        from assetra.units import DemandUnit, StaticUnit, HydroUnit

        hydro2 = HydroUnit(id=2, nameplate_capacity=3,
                           monthly_expected_generation=xr.DataArray([3.0] * 12, coords=dict(month=list(range(1, 13)))))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([12, 14, 9])),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10] * 3)),
            hydro2,
        )
        return e1, self._simulation(3, 1), {"hydro2": self._system(hydro2)}

    def _f7_two_hydro(self):
        """Hydro, two units: Hydro 2 dispatches before Hydro 3, which is the resource."""
        from assetra.units import DemandUnit, StaticUnit, HydroUnit

        hydro3 = HydroUnit(id=3, nameplate_capacity=3,
                           monthly_expected_generation=xr.DataArray([1.9] * 12, coords=dict(month=list(range(1, 13)))))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([12, 14, 9])),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10] * 3)),
            HydroUnit(id=2, nameplate_capacity=1.2,
                      monthly_expected_generation=xr.DataArray([3.0] * 12, coords=dict(month=list(range(1, 13))))),
            hydro3,
        )
        return e1, self._simulation(3, 1), {"hydro3": self._system(hydro3)}

    def _f10(self):
        """Negative accreditation of storage: 7 hours, 1 trial, Storage 2 (the resource) dispatches before Storage 3."""
        from assetra.units import DemandUnit, StaticUnit, StorageUnit

        storage2 = StorageUnit(id=2, nameplate_capacity=2.25, charge_rate=2.25, discharge_rate=2.25,
                               charge_capacity=4.5, roundtrip_efficiency=1.0, initial_soc=0)
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([3] * 7)),
            StaticUnit(id=1, nameplate_capacity=6,
                       hourly_capacity=get_sample_time_series([4.75, 4.25, 3.75, 0.25, 2.5, 1.25, 5.75])),
            storage2,
            StorageUnit(id=3, nameplate_capacity=0.5, charge_rate=0.5, discharge_rate=0.5, charge_capacity=2,
                        roundtrip_efficiency=1.0, initial_soc=1),
        )
        return e1, self._simulation(7, 1), {"storage2": self._system(storage2)}

    def test_resource_unit_ids(self):
        from assetra.contribution import _resource_unit_ids
        from assetra.units import StochasticUnit
        e1, ps, r = self._f3()
        self.assertEqual({StochasticUnit: [2]}, _resource_unit_ids(e1, r["g"]))
        self.assertEqual({StochasticUnit: [2, 3]}, _resource_unit_ids(e1, r["gh"]))

    def test_resource_unit_ids_errors(self):
        from assetra.contribution import _resource_unit_ids
        from assetra.system import EnergySystem
        from assetra.units import DemandUnit, StaticUnit, StochasticUnit, HydroUnit
        e1, ps, r = self._f1()
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(TypeError, "EnergySystem"):
            _resource_unit_ids(e1, "static2")
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "empty"):
            _resource_unit_ids(e1, EnergySystem({}))
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "not supported"):
            _resource_unit_ids(e1, self._system(DemandUnit(id=0, hourly_demand=get_sample_time_series([110, 120, 80, 70]))))
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "not in the base system"):
            _resource_unit_ids(e1, self._system(StaticUnit(id=9, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1] * 4))))
        # a distinct but equal object matches; the same id under another type does not
        equal = self._system(StaticUnit(id=2, nameplate_capacity=10, hourly_capacity=get_sample_time_series([5, 10, 0, 10])))
        self.assertEqual({StaticUnit: [2]}, _resource_unit_ids(e1, equal))
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "not in the base system"):
            _resource_unit_ids(e1, self._system(StochasticUnit(id=2, nameplate_capacity=10,
                hourly_capacity=get_sample_time_series([10] * 4), hourly_forced_outage_rate=get_sample_time_series([0] * 4))))
        e7, ps7, r7 = self._f7()
        no_ids = EnergySystem({**e7.unit_datasets, HydroUnit: e7.unit_datasets[HydroUnit].drop_vars("energy_unit")})
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "unit ids"):
            _resource_unit_ids(no_ids, r7["hydro2"])
        # the resources' own hydro dataset without ids raises as well, against a base system that has them
        resources_no_ids = EnergySystem({HydroUnit: r7["hydro2"].unit_datasets[HydroUnit].drop_vars("energy_unit")})
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "unit ids"):
            _resource_unit_ids(e7, resources_no_ids)

    def test_replay_static(self):
        import numpy as np
        from assetra.contribution import _replay_non_responsive, _resource_unit_ids
        e1, ps, r = self._f1()
        net, output = _replay_non_responsive(e1, _resource_unit_ids(e1, r["static2"]), ps)
        self.assertEqual(("trial", "time"), net.dims)
        np.testing.assert_allclose(net.values, [[-5, -10, 20, 40]] * 2)
        np.testing.assert_allclose(output.values, [[5, 10, 0, 10]] * 2)
        self.assertIn("no system assigned", repr(ps))  # the user's simulation is untouched

    def test_replay_stochastic_own_draws(self):
        import numpy as np
        from assetra.contribution import _replay_non_responsive, _resource_unit_ids
        e1, ps, r = self._f3()
        net, output = _replay_non_responsive(e1, _resource_unit_ids(e1, r["g"]), ps)
        np.testing.assert_allclose(net.values, [[-1, -10, 20, 40]] * 3)
        np.testing.assert_allclose(output.values, [[4, 0, 0, 0]] * 3)

    def test_replay_hydro_sequential(self):
        import numpy as np
        from assetra.contribution import _replay_non_responsive, _resource_unit_ids
        e1, ps, r = self._f7_two_hydro()
        net, output = _replay_non_responsive(e1, _resource_unit_ids(e1, r["hydro3"]), ps)
        np.testing.assert_allclose(net.values, [[-0.5, -1.4, 1.0]], atol=1e-9)
        np.testing.assert_allclose(output.values, [[0.5, 1.4, 0.0]], atol=1e-9)

    def test_scale_storage_dataset(self):
        from assetra.contribution import _scale_storage_dataset
        from assetra.units import StorageUnit
        e1, ps, r = self._f5()
        ds = e1.unit_datasets[StorageUnit]
        before = ds.copy(deep=True)
        scaled = _scale_storage_dataset(ds, [2], 0.1, 1.0)
        self.assertTrue(ds.identical(before))
        ratings = ("charge_rate", "discharge_rate", "charge_capacity", "nameplate_capacity")
        for v in ratings:
            self.assertAlmostEqual(1.1, float(scaled[v].sel(energy_unit=2)), delta=1e-12, msg=v)
        self.assertTrue(scaled["initial_soc"].identical(ds["initial_soc"]))
        self.assertNotIn("initial_soc", _scale_storage_dataset(ds.drop_vars("initial_soc"), [2], 0.1, 1.0))
        # k = 0 empties the units exactly, also for values that are not exact in binary
        emptied = _scale_storage_dataset(ds.assign({v: ds[v] * 54.8 for v in ratings}), [2], -82.07, 82.07)
        for v in ratings:
            self.assertEqual(0.0, float(emptied[v].sel(energy_unit=2)), msg=v)

    def test_mri_constructor_errors(self):
        from assetra.metrics import ExpectedUnservedEnergy, LossOfLoadFrequency
        from assetra.contribution import MarginalReliabilityImpact

        def rc(e1, ps, method, step=None, metric=ExpectedUnservedEnergy):
            return MarginalReliabilityImpact(e1, ps, metric, perturbation_size=step, method=method)

        e1, ps, r = self._f1()
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(TypeError, "EnergySystem"):
            rc("x", ps, "forward")
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(TypeError, "ResourceAdequacyMetric"):
            rc(e1, ps, "forward", metric=ExpectedUnservedEnergy(ps))
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "method"):
            rc(e1, ps, "newton")
        for step in (0, -1, float("nan")):
            with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "perturbation_size", msg=repr(step)):
                rc(e1, ps, "forward", step)
        # MRI accepts any metric class
        self.assertEqual(0, rc(e1, ps, "forward", metric=LossOfLoadFrequency).perturbation_count)

    def test_mri_values(self):
        import math
        import numpy as np
        from assetra.metrics import ExpectedUnservedEnergy as EUE, LossOfLoadHours as LOLH
        from assetra.contribution import MarginalReliabilityImpact

        def rc(e1, ps, method, step=None, metric=EUE):
            return MarginalReliabilityImpact(e1, ps, metric, perturbation_size=step, method=method)

        # (builder, resource key, step, metric, forward, central)
        cases = [
            (self._f1, "static2", 1, EUE, 0.75, 0.75), (self._f1, "static2", None, EUE, 0.75, 0.75),
            (self._f1, "fleet", 1, EUE, 43 / 44, 43 / 44),  # the whole static fleet
            (self._f2, "static2", 1, EUE, 0.5, 1 / 3),
            (self._f3, "g", 0.5, EUE, 0.5, 0.5), (self._f3, "g", 4, EUE, 0.2, 5 / 13),
            (self._f3, "h", 1, EUE, 1.0, 1.0), (self._f3, "gh", 0.5, EUE, 2 / 3, 2 / 3),
            (self._f3, "h", 1, LOLH, 1.0, 1.0),
            (self._f4, "stoch2", 1, EUE, 0.0, 0.0),
            (self._f5, "storage2", 0.1, EUE, 0.5, 0.5), (self._f5, "storage2", 1, EUE, 1.0, 2 / 3),
            (self._f5, "static1", 0.1, EUE, 0.5, 0.5), (self._f5, "hybrid", 0.2, EUE, 0.5, 0.5),
            (self._f6, "storage2", 0.1, EUE, 2.0, 2.0),
            (self._f7, "hydro2", 0.3, EUE, 0.5, 0.5), (self._f7_two_hydro, "hydro3", 0.3, EUE, 19 / 60, 19 / 60),
            (self._f10, "storage2", 0.225, EUE, -0.9, -9 / 64),
        ]
        for case in cases:
            builder, key, step, metric, forward, central = case
            e1, ps, r = builder()
            for method, expected, count in (("forward", forward, 1), ("central", central, 2)):
                mri = rc(e1, ps, method, step, metric)
                # F4 gives 0 for any draw with an outage; the seed only rules out an outage-free draw
                np.random.seed(0)
                observed = mri.evaluate(r[key])
                self.assertAlmostEqual(expected, observed, delta=0.01, msg=f"{case} {method}")
                self.assertEqual(count, mri.perturbation_count, msg=f"{case} {method}")
                if expected == 0:  # no effect: an exact 0 is 0.0, never -0.0 (copysign reads the sign of a zero)
                    self.assertEqual(1.0, math.copysign(1.0, observed), msg=f"{case} {method}")

    def test_mri_errors(self):
        from assetra.units import StaticUnit
        from assetra.metrics import ExpectedUnservedEnergy, LossOfLoadDays, LossOfLoadHours
        from assetra.contribution import MarginalReliabilityImpact

        def rc(e1, ps, method, step=None, metric=ExpectedUnservedEnergy):
            return MarginalReliabilityImpact(e1, ps, metric, perturbation_size=step, method=method)

        e1, ps, r = self._f1()
        for metric in (LossOfLoadHours, LossOfLoadDays):
            for method in ("forward", "central"):
                with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "does not respond", msg=f"{metric.__name__} {method}"):
                    rc(e1, ps, method, 1, metric).evaluate(r["static2"])
        e1, ps, r = self._f1(demand=(50, 50, 50, 50))
        for method in ("forward", "central"):
            with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "no risk", msg=method):
                rc(e1, ps, method).evaluate(r["static2"])
        e1, ps, r = self._f3()
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "perturbation_size"):
            rc(e1, ps, "central", 5).evaluate(r["g"])
        static3 = StaticUnit(id=3, nameplate_capacity=0, hourly_capacity=get_sample_time_series([0] * 4))
        e1, ps, r = self._f1(extra_units=(static3,))
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "nameplate"):
            MarginalReliabilityImpact(e1, ps, ExpectedUnservedEnergy).evaluate(self._system(static3))
        static3 = StaticUnit(id=3, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1, -1, 0, 0]))
        e1, ps, r = self._f1(extra_units=(static3,))
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "negative"):
            MarginalReliabilityImpact(e1, ps, ExpectedUnservedEnergy).evaluate(self._system(static3))

    def test_mri_does_not_mutate_inputs(self):
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import MarginalReliabilityImpact
        e1, ps, r = self._f5()
        resources = r["storage2"]
        copies = [(system, {t: ds.copy(deep=True) for t, ds in system.unit_datasets.items()}) for system in (e1, resources)]
        MarginalReliabilityImpact(e1, ps, ExpectedUnservedEnergy, perturbation_size=0.1, method="central").evaluate(resources)
        for system, datasets in copies:
            self.assertEqual(list(datasets), list(system.unit_datasets))
            for unit_type, dataset in datasets.items():
                self.assertTrue(system.unit_datasets[unit_type].identical(dataset), msg=unit_type.__name__)
        self.assertIn("no system assigned", repr(ps))

    def test_mri_loaded_and_trimmed_systems(self):
        import tempfile
        import warnings
        from pathlib import Path
        from assetra.system import EnergySystem
        from assetra.units import StorageUnit
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import MarginalReliabilityImpact

        def rc(e1, ps, method, step=None, metric=ExpectedUnservedEnergy):
            return MarginalReliabilityImpact(e1, ps, metric, perturbation_size=step, method=method)

        e1, ps, r = self._f1()
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            # third-party noise: NumPy >= 2.5 deprecation inside xarray's netCDF4 writer
            warnings.filterwarnings("ignore", "Setting the shape on a NumPy array", DeprecationWarning)
            e1.save(Path(tmp, "system"))
            loaded = EnergySystem()
            loaded.load(Path(tmp, "system"))
        self.assertAlmostEqual(0.75, rc(loaded, ps, "forward", 1).evaluate(r["static2"]), delta=0.01)
        # a storage dataset saved before initial_soc existed
        e1, ps, r = self._f5()
        old = EnergySystem({**e1.unit_datasets, StorageUnit: e1.unit_datasets[StorageUnit].drop_vars("initial_soc")})
        self.assertAlmostEqual(0.5, rc(old, ps, "central", 0.1).evaluate(r["storage2"]), delta=0.01)
        # a simulation window shorter than the unit profiles
        e1, ps, r = self._f1(extra_hours=2)
        self.assertAlmostEqual(0.75, rc(e1, ps, "forward", 1).evaluate(r["static2"]), delta=0.01)

    def test_mri_reuse_after_error(self):
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import MarginalReliabilityImpact
        e1, ps, r = self._f3()
        rc = MarginalReliabilityImpact(e1, ps, ExpectedUnservedEnergy, perturbation_size=5, method="central")
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "perturbation_size"):
            rc.evaluate(r["g"])
        self.assertAlmostEqual(7 / 12, rc.evaluate(r["gh"]), delta=0.01)
        self.assertEqual(2, rc.perturbation_count)

    def test_mri_custom_types_and_seeds(self):
        import numpy as np
        from assetra.units import NONRESPONSIVE_UNIT_TYPES, RESPONSIVE_UNIT_TYPES, DemandUnit, StaticUnit, StochasticUnit
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import MarginalReliabilityImpact

        def rc(e1, ps, method, step=None, metric=ExpectedUnservedEnergy):
            return MarginalReliabilityImpact(e1, ps, metric, perturbation_size=step, method=method)

        class DummyUnit(StaticUnit):
            pass

        NONRESPONSIVE_UNIT_TYPES.append(DummyUnit)
        RESPONSIVE_UNIT_TYPES.append(DummyUnit)
        try:
            e1, ps, r = self._f3()
            self.assertAlmostEqual(0.5, rc(e1, ps, "forward", 0.5).evaluate(r["g"]), delta=0.01)
            e1, ps, r = self._f5()
            self.assertAlmostEqual(0.5, rc(e1, ps, "forward", 0.1).evaluate(r["storage2"]), delta=0.01)
        finally:
            NONRESPONSIVE_UNIT_TYPES.remove(DummyUnit)
            RESPONSIVE_UNIT_TYPES.remove(DummyUnit)

        # every call samples afresh, so the same seed gives the same factor
        stoch2 = StochasticUnit(id=2, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1] * 4),
                                hourly_forced_outage_rate=get_sample_time_series([0.5] * 4))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([12] * 4)),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10] * 4)),
            stoch2,
            StochasticUnit(id=3, nameplate_capacity=4, hourly_capacity=get_sample_time_series([4] * 4),
                           hourly_forced_outage_rate=get_sample_time_series([0.5] * 4)),
        )
        mri = rc(e1, self._simulation(4, 20), "forward", 0.5)
        observed = []
        for _ in range(2):
            np.random.seed(1)
            observed.append(mri.evaluate(self._system(stoch2)))
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(1, mri.perturbation_count)  # reset by every call

    def test_mri_exact_ties_integer_data(self):
        from assetra.units import DemandUnit, StaticUnit, StochasticUnit
        from assetra.metrics import ExpectedUnservedEnergy, LossOfLoadHours
        from assetra.contribution import MarginalReliabilityImpact
        # F3 with h at nameplate and capacity 49 and demand raised by 47, so N_nr is still [-1, -10, 20, 40]:
        # as in F3, a step of 1 clears hour 0's deficit exactly in both M(+1) and P(+1), so every factor is 1.0.
        # The step must be applied multiply-first: 1 / 49 * 49 rounds to 0.9999999999999999.
        h = StochasticUnit(id=3, nameplate_capacity=49, hourly_capacity=get_sample_time_series([49] * 4),
                           hourly_forced_outage_rate=get_sample_time_series([0, 0, 1, 1]))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([104, 109, 30, 10])),
            StaticUnit(id=1, nameplate_capacity=50, hourly_capacity=get_sample_time_series([50] * 4)),
            StochasticUnit(id=2, nameplate_capacity=4, hourly_capacity=get_sample_time_series([4] * 4),
                           hourly_forced_outage_rate=get_sample_time_series([0, 1, 1, 1])),
            h,
        )
        ps = self._simulation(4, 3)
        for metric in (LossOfLoadHours, ExpectedUnservedEnergy):
            for method in ("forward", "central"):
                mri = MarginalReliabilityImpact(e1, ps, metric, perturbation_size=1, method=method)
                self.assertAlmostEqual(1.0, mri.evaluate(self._system(h)), delta=0.01, msg=f"{metric.__name__} {method}")

    def test_mri_exact_ties_integer_storage(self):
        from assetra.units import DemandUnit, StaticUnit, StorageUnit
        from assetra.metrics import LossOfLoadHours
        from assetra.contribution import MarginalReliabilityImpact
        # Storage 2 is the resource (x_T = 13): rates 13, capacity 13, full; N_nr = [-15, -15]; step 2, LOLH.
        # M0: discharge 13 -> nets [-2, -15] -> 2. M(+2), k = 15/13: rates and energy 15 clear hour 0 exactly
        # -> [0, -15] -> 1. P(+2): [-13, -13] -> [0, -13] -> 1. M(-2), k = 11/13: [-4, -15] -> 2.
        # P(-2): [-17, -17] -> [-4, -17] -> 2. Forward (1 - 2) / (1 - 2) = 1.0; central (1 - 2) / (1 - 2) = 1.0.
        # The ratings must be scaled multiply-first: 13 * (1 + 2 / 13) rounds to 14.999999999999998.
        storage2 = StorageUnit(id=2, nameplate_capacity=13, charge_rate=13, discharge_rate=13, charge_capacity=13,
                               roundtrip_efficiency=1.0, initial_soc=1)
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([25, 25])),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10, 10])),
            storage2,
        )
        ps = self._simulation(2, 1)
        for method in ("forward", "central"):
            mri = MarginalReliabilityImpact(e1, ps, LossOfLoadHours, perturbation_size=2, method=method)
            self.assertAlmostEqual(1.0, mri.evaluate(self._system(storage2)), delta=0.01, msg=method)

    def test_root_linear(self):
        from assetra.contribution import _find_largest_root

        def g(c):
            return 2 * c - 1.5

        # guaranteed bracket [0, 1]: bisection evaluates neither end, the secant evaluates both and its first step is the root
        root, evaluations = _find_largest_root(g, 1, 0.01, "bisection", True, 1.5e-8, True)
        self.assertAlmostEqual(0.75, root, delta=0.005, msg="bisection")
        self.assertEqual(7, evaluations, msg="bisection")
        root, evaluations = _find_largest_root(g, 1, 0.01, "secant", True, 1.5e-8, True)
        self.assertAlmostEqual(0.75, root, places=12, msg="secant")
        self.assertEqual(3, evaluations, msg="secant")

    def test_root_flat(self):
        from assetra.contribution import _find_largest_root

        for method in ("bisection", "secant"):
            with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "does not respond", msg=method):
                _find_largest_root(lambda c: 0.0, 1, 0.01, method, False, 0.0, False)

    def test_root_plateau(self):
        from assetra.contribution import _find_largest_root

        def g(c):  # below 0 up to c = 0, a plateau of 0 up to c = 20, above 0 beyond
            return -1.0 if c <= 0 else (0.0 if c <= 20 else 1.0)

        # g(1) is on the plateau: the search stops there and does not cross it
        for method in ("bisection", "secant"):
            root, evaluations = _find_largest_root(g, 1, 0.01, method, False, 0.0, False)
            self.assertAlmostEqual(1.0, root, delta=1e-12, msg=method)
            self.assertEqual(2, evaluations, msg=method)

    def test_root_upward_expansion(self):
        from assetra.contribution import _find_largest_root

        def g(c):
            return 2 * c - 0.4

        # g(0.1) < 0, so the upper end doubles to 0.2, where g is 0: the root, found with the third evaluation
        for method in ("bisection", "secant"):
            root, evaluations = _find_largest_root(g, 0.1, 0.001, method, False, 2e-9, True)
            self.assertAlmostEqual(0.2, root, delta=1e-12, msg=method)
            self.assertEqual(3, evaluations, msg=method)

    def test_root_downward_expansion(self):
        from assetra.contribution import _find_largest_root

        def g(c):
            return 2 * c + 0.09

        # g(0) > 0, so the lower end moves down to -0.225 and the bracket is [-0.225, 0]. Bisection halves it 7 times
        # (0.225 / 0.00225 = 100 < 2 ** 7), after 3 evaluations; the secant through the last two points is the root
        for method, expected_evaluations in (("bisection", 10), ("secant", 4)):
            root, evaluations = _find_largest_root(g, 0.225, 0.00225, method, False, 1e-10, True)
            self.assertAlmostEqual(-0.045, root, delta=0.00113, msg=method)
            self.assertEqual(expected_evaluations, evaluations, msg=method)

    def test_root_expansion_cap(self):
        from assetra.contribution import _find_largest_root, _MAX_DOUBLINGS, _MAX_ITERATIONS

        def recorded(g, calls):
            def recording_g(c):
                calls.append(c)
                return g(c)

            return recording_g

        # g stays below 0 however far the upper end doubles, or above 0 however far the lower end moves down: both ends
        # are evaluated, then _MAX_DOUBLINGS expansions (the upper end doubles, the lower end moves down by 1, 2, 4, ...)
        doublings = range(1, _MAX_DOUBLINGS + 1)
        for method in ("bisection", "secant"):
            for value, expansions in ((-1.0, [2**k for k in doublings]), (1.0, [1 - 2**k for k in doublings])):
                calls = []
                with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "bracket", msg=f"{method} {value}"):
                    _find_largest_root(recorded(lambda c: value, calls), 1, 0.01, method, False, 0.0, False)
                self.assertEqual([0, 1] + expansions, calls, msg=f"{method} {value}")

        # the secant also gives up, after _MAX_ITERATIONS steps, when the bracket never closes: with the root at 0, where
        # floating-point numbers are finest, the bracket [0, 2**-k] can still be split and a zero tolerance is out of reach
        calls = []
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "converge"):
            _find_largest_root(recorded(lambda c: 2 * c, calls), 1, 0.0, "secant", False, 0.0, False)
        self.assertEqual(2 + _MAX_ITERATIONS, len(calls))

    def test_root_secant_steep_convex(self):
        from assetra.contribution import _find_largest_root

        def g(c):
            return 0.01 * (c - 0.5) if c < 0.5 else 100 * (c - 0.5)

        # the secant finds the root of a steep convex g (slopes of 0.01 below it and 100 above it) within the tolerance,
        # in at most twice the bisection's evaluations plus 2. The bound is loose: the secant meets it without its halving
        # safeguard too, which test_root_secant_halving_safeguard pins
        for method in ("bisection", "secant"):
            root, evaluations = _find_largest_root(g, 10, 0.1, method, True, 0.0, False)
            self.assertAlmostEqual(0.5, root, delta=0.1, msg=method)
            if method == "bisection":
                self.assertEqual(7, evaluations, msg=method)
            else:
                self.assertLessEqual(evaluations, 16, msg=method)

    def test_root_secant_halving_safeguard(self):
        from assetra.contribution import _find_largest_root

        def g(c):  # a triple root at 1.5, where the secant converges slowly
            return (c - 1.5) ** 3

        # bisection halves [0, 10] 14 times to a width below 1e-3. The secant takes a bisection step whenever its previous
        # step did not halve the bracket, which bounds its cost at twice the bisection's plus the 2 end evaluations:
        # 2 * 14 + 2 = 30. It takes 20, and 32 without that rule
        root, evaluations = _find_largest_root(g, 10, 1e-3, "bisection", True, 0.0, False)
        self.assertAlmostEqual(1.5, root, delta=1e-3, msg="bisection")
        self.assertEqual(14, evaluations, msg="bisection")
        root, evaluations = _find_largest_root(g, 10, 1e-3, "secant", True, 0.0, False)
        self.assertAlmostEqual(1.5, root, delta=1e-3, msg="secant")
        self.assertLessEqual(evaluations, 2 * 14 + 2, msg="secant")

    def test_root_float_resolution(self):
        from assetra.contribution import _find_largest_root, _MAX_ITERATIONS

        def g(c):
            calls.append(c)
            if len(calls) > 1000:  # a search that does not stop would hang the suite
                self.fail("the search did not stop at floating-point resolution")
            return 2 * c - 1.5

        # a tolerance below the spacing of floating-point numbers around the root cannot be met: once the bracket cannot be
        # split (its midpoint is not strictly inside it), both methods return that midpoint instead of looping or raising
        for method, limit in (("bisection", 70), ("secant", _MAX_ITERATIONS + 2)):
            calls = []
            root, evaluations = _find_largest_root(g, 1, 1e-20, method, True, 0.0, False)
            self.assertAlmostEqual(0.75, root, delta=1e-12, msg=method)
            self.assertLessEqual(evaluations, limit, msg=method)

    def test_marginal_elcc_policy_errors(self):
        from assetra.metrics import ExpectedUnservedEnergy, LossOfLoadFrequency
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability

        def rc(e1, ps, method, step=None, metric=ExpectedUnservedEnergy):
            return MarginalEffectiveLoadCarryingCapability(e1, ps, metric, perturbation_size=step, method=method)

        class _EUE(ExpectedUnservedEnergy):
            pass

        # the root search needs risk that never decreases as load is added: exactly EUE, LOLH or LOLD, no subclass
        e1, ps, r = self._f1()
        for metric in (LossOfLoadFrequency, _EUE):
            with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "monotone", msg=metric.__name__):
                rc(e1, ps, "bisection", metric=metric)
        for pct in (0, 1):
            with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "additional_demand_resolution_pct", msg=repr(pct)):
                rc(e1, ps, "bisection").evaluate(r["static2"], pct)
        e1, ps, r = self._f1(demand=(50, 50, 50, 50))
        for method in ("bisection", "secant"):
            with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "no risk", msg=method):
                rc(e1, ps, method).evaluate(r["static2"])

    def test_marginal_elcc_values(self):
        import math
        import numpy as np
        from assetra.metrics import ExpectedUnservedEnergy as EUE, LossOfLoadHours as LOLH
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability

        def rc(e1, ps, method, step=None, metric=EUE):
            return MarginalEffectiveLoadCarryingCapability(e1, ps, metric, perturbation_size=step, method=method)

        # (builder, resource key, step, metric, factor of both methods)
        cases = [
            (self._f1, "static2", 1, EUE, 0.75), (self._f1, "static2", None, EUE, 0.75),
            (self._f1, "fleet", 1, EUE, 43 / 44),  # the whole static fleet
            (self._f2, "static2", 1, EUE, 0.25),
            (self._f3, "g", 0.5, EUE, 0.5), (self._f3, "g", 4, EUE, 0.25),
            (self._f3, "h", 1, EUE, 1.0), (self._f3, "gh", 0.5, EUE, 2 / 3),
            (self._f3, "h", 1, LOLH, 1.0),  # a plateau of baseline risk at the step, which is not crossed
            (self._f4, "stoch2", 1, EUE, 0.0),
            (self._f5, "storage2", 0.1, EUE, 0.5), (self._f5, "storage2", 1, EUE, 0.5),
            (self._f5, "static1", 0.1, EUE, 0.5), (self._f5, "hybrid", 0.2, EUE, 0.5),
            (self._f6, "storage2", 0.1, EUE, 2.0),  # the bracket expands upward
            (self._f7, "hydro2", 0.3, EUE, 0.5), (self._f7_two_hydro, "hydro3", 0.3, EUE, 19 / 60),
            (self._f10, "storage2", 0.225, EUE, -0.2),  # the bracket expands downward
        ]
        for case in cases:
            builder, key, step, metric, expected = case
            e1, ps, r = builder()
            for method in ("bisection", "secant"):
                elcc = rc(e1, ps, method, step, metric)
                # F4 gives 0 for any draw with an outage; the seed only rules out an outage-free draw
                np.random.seed(0)
                observed = elcc.evaluate(r[key])
                self.assertAlmostEqual(expected, observed, delta=0.01, msg=f"{case} {method}")
                if expected == 0:  # F4's secant ends on an exact root at 0, which is 0.0, never -0.0
                    self.assertEqual(1.0, math.copysign(1.0, observed), msg=f"{case} {method}")

    def test_marginal_elcc_counts(self):
        import numpy as np
        from assetra.metrics import ExpectedUnservedEnergy as EUE, LossOfLoadHours as LOLH
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability

        def rc(e1, ps, method, step=None, metric=EUE):
            return MarginalEffectiveLoadCarryingCapability(e1, ps, metric, perturbation_size=step, method=method)

        # (builder, resource key, step, metric, method, perturbation count)
        cases = [
            # F1's bracket [0, step] is guaranteed: bisection halves it 7 times without evaluating its ends; the secant
            # evaluates both ends, and its first step is the exact root, since g is linear
            (self._f1, "static2", 1, EUE, "bisection", 7), (self._f1, "static2", 1, EUE, "secant", 3),
            (self._f1, "static2", None, EUE, "bisection", 7), (self._f1, "static2", None, EUE, "secant", 3),
            (self._f3, "h", 1, EUE, "secant", 2),  # the exact root at the step
            # a plateau of baseline risk at the step
            (self._f3, "h", 1, LOLH, "bisection", 2), (self._f3, "h", 1, LOLH, "secant", 2),
            (self._f4, "stoch2", 1, EUE, "secant", 2),  # the exact root at 0
        ]
        for case in cases:
            builder, key, step, metric, method, expected = case
            e1, ps, r = builder()
            elcc = rc(e1, ps, method, step, metric)
            np.random.seed(0)
            elcc.evaluate(r[key])
            self.assertEqual(expected, elcc.perturbation_count, msg=str(case))

    def test_marginal_elcc_flat_metric(self):
        from assetra.metrics import LossOfLoadDays, LossOfLoadHours
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability
        # under LOLH and LOLD, F1's step of 1 changes no shortfall hour or day, so g is 0 at both ends; both evaluations are counted
        e1, ps, r = self._f1()
        for metric in (LossOfLoadHours, LossOfLoadDays):
            for method in ("bisection", "secant"):
                elcc = MarginalEffectiveLoadCarryingCapability(e1, ps, metric, perturbation_size=1, method=method)
                with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(RuntimeWarning, "does not respond", msg=f"{metric.__name__} {method}"):
                    elcc.evaluate(r["static2"])
                self.assertEqual(2, elcc.perturbation_count, msg=f"{metric.__name__} {method}")

    def test_marginal_elcc_largest_root_of_a_plateau(self):
        from assetra.metrics import LossOfLoadHours
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability

        # F7 under LOLH with a step of 2: g is 0 on [0, 1] and 1 beyond, so the largest load with g <= 0 is 1 and the factor
        # is 1 / 2. Ties (g = 0) must move the lower end of the bracket, or the search settles near 0 (a factor of 0.004).
        # The secant meets equal values of g beyond the root (both 1), where it must take a bisection step instead of
        # dividing by 0. Both ends of [0, 2] are evaluated, and 7 halvings bring its width (2 / 2 ** 7) below the resolution
        # of 0.02: 9 evaluations for both methods
        e1, ps, r = self._f7()
        for method in ("bisection", "secant"):
            elcc = MarginalEffectiveLoadCarryingCapability(e1, ps, LossOfLoadHours, perturbation_size=2, method=method)
            self.assertAlmostEqual(0.5, elcc.evaluate(r["hydro2"]), delta=0.01, msg=method)
            self.assertEqual(9, elcc.perturbation_count, msg=method)

    def test_output_above_nameplate(self):
        from assetra.units import DemandUnit, StaticUnit
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability, MarginalReliabilityImpact

        # F1 with a demand of 130 in hour 1, where the resource (nameplate capacity 10) produces 20: N_nr = [-5, -10, 20, 40],
        # and its availability in the two shortfall hours is 0.5 and 2, mean 1.25, so every method gives 1.25 and
        # g(c) = 2 c - 2.5. With output above its nameplate capacity, the bracket [0, step] does not hold the root (g(1) < 0),
        # so the ELCC must evaluate the ends of its bracket instead of assuming them (bisection would return 0.996).
        # ELCC bisection takes 2 ends + 1 upward expansion + 7 halvings = 10 evaluations, and the secant 4
        static2 = StaticUnit(id=2, nameplate_capacity=10, hourly_capacity=get_sample_time_series([5, 20, 0, 10]))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([110, 130, 80, 70])),
            StaticUnit(id=1, nameplate_capacity=100, hourly_capacity=get_sample_time_series([100] * 4)),
            static2,
        )
        ps = self._simulation(4, 2)
        resources = self._system(static2)
        # (metric class, method, perturbation count)
        cases = [
            (MarginalReliabilityImpact, "forward", 1), (MarginalReliabilityImpact, "central", 2),
            (MarginalEffectiveLoadCarryingCapability, "bisection", 10), (MarginalEffectiveLoadCarryingCapability, "secant", 4),
        ]
        for metric, method, count in cases:
            accreditation = metric(e1, ps, ExpectedUnservedEnergy, perturbation_size=1, method=method)
            self.assertAlmostEqual(1.25, accreditation.evaluate(resources), delta=0.01, msg=f"{metric.__name__} {method}")
            self.assertEqual(count, accreditation.perturbation_count, msg=f"{metric.__name__} {method}")

    def test_marginal_elcc_inputs_unchanged_and_reusable(self):
        import numpy as np
        from assetra.units import DemandUnit, StaticUnit, StochasticUnit
        from assetra.metrics import ExpectedUnservedEnergy
        from assetra.contribution import MarginalEffectiveLoadCarryingCapability

        def rc(e1, ps, method, step=None, metric=ExpectedUnservedEnergy):
            return MarginalEffectiveLoadCarryingCapability(e1, ps, metric, perturbation_size=step, method=method)

        # neither solver modifies the systems, their datasets or the simulation
        e1, ps, r = self._f5()
        resources = r["storage2"]
        copies = [(system, {t: ds.copy(deep=True) for t, ds in system.unit_datasets.items()}) for system in (e1, resources)]
        for method in ("bisection", "secant"):
            rc(e1, ps, method, 0.1).evaluate(resources)
            for system, datasets in copies:
                self.assertEqual(list(datasets), list(system.unit_datasets), msg=method)
                for unit_type, dataset in datasets.items():
                    self.assertTrue(system.unit_datasets[unit_type].identical(dataset), msg=f"{unit_type.__name__} {method}")
            self.assertIn("no system assigned", repr(ps), msg=method)

        # one object reused after a call that raised gives an independent result and counts only the latest call
        e1, ps, r = self._f3()
        elcc = rc(e1, ps, "secant", 1)
        with self.assertLogs("assetra.contribution", "ERROR"), self.assertRaisesRegex(ValueError, "additional_demand_resolution_pct"):
            elcc.evaluate(r["h"], additional_demand_resolution_pct=0)
        self.assertAlmostEqual(1.0, elcc.evaluate(r["h"]), delta=0.01)
        self.assertEqual(2, elcc.perturbation_count)

        # every call samples afresh, so the same seed gives the same factor
        stoch2 = StochasticUnit(id=2, nameplate_capacity=1, hourly_capacity=get_sample_time_series([1] * 4),
                                hourly_forced_outage_rate=get_sample_time_series([0.5] * 4))
        e1 = self._system(
            DemandUnit(id=0, hourly_demand=get_sample_time_series([12] * 4)),
            StaticUnit(id=1, nameplate_capacity=10, hourly_capacity=get_sample_time_series([10] * 4)),
            stoch2,
            StochasticUnit(id=3, nameplate_capacity=4, hourly_capacity=get_sample_time_series([4] * 4),
                           hourly_forced_outage_rate=get_sample_time_series([0.5] * 4)),
        )
        elcc = rc(e1, self._simulation(4, 20), "secant", 0.5)
        observed, counts = [], []
        for _ in range(2):
            np.random.seed(1)
            observed.append(elcc.evaluate(self._system(stoch2)))
            counts.append(elcc.perturbation_count)
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(counts[0], counts[1])  # reset by every call

#new -- hydrounit tests 
from assetra.units import HydroUnit
import pandas as pd
import numpy as np

def create_test_data(months=12, hours_per_day=24):
    """
    Generate test data for hydropower unit simulation.

    Creates synthetic time series data for testing hydro unit dispatch functions. Returns:
        - net_hourly_capacity: Time series (hourly) of system's net available capacity, as a DataArray with random values from -10 to 0 for each hour.
        - monthly_expected_generation: A monthly array with a constant value (100) for each month.
        - hourly_forced_outage_rate: Time series (hourly) of forced outage rates (set at 0.1 for every hour).

    Args:
        months (int): Number of months for which the test data is generated.
        hours_per_day (int): Number of hours in a day (default 24).

    Returns:
        net_hourly_capacity (xr.DataArray): Array of shape (time,), hourly, with random values from -10 to 0.
        monthly_expected_generation (xr.DataArray): Array of shape (month,), with value 100.
        hourly_forced_outage_rate (xr.DataArray): Array of shape (time,), with value 0.1.
    
    Example returned shapes for default arguments:
        net_hourly_capacity: (months * hours_per_day * 30,)  # hours for 30 days per month
        monthly_expected_generation: (months,)
        hourly_forced_outage_rate: (months * hours_per_day * 30,)
    """

    times = pd.date_range(start='2021-01-01', periods=months*hours_per_day*30, freq='h')
    np.random.seed(0)  # Set seed for reproducibility
    net_hourly_capacity = xr.DataArray(
        np.random.uniform(low=-10, high=0, size=len(times)), 
        dims='time', 
        coords={'time': times}
    )
    
    month_coords = range(1, months + 1)
    monthly_expected_generation = xr.DataArray(
        [100] * months,
        dims='month',
        coords={'month': month_coords}
    )
    
    hourly_forced_outage_rate = xr.DataArray(
        np.zeros(len(times)) + 0.1, 
        dims='time',
        coords={'time': times}
    )

    return net_hourly_capacity, monthly_expected_generation, hourly_forced_outage_rate

class TestHydroUnit(unittest.TestCase):
    from assetra.units import HydroUnit
    def test_hydro_unit_building(self):
        """
        Test HydroUnit initialization and attribute assignment.

        Ensures:
            - The HydroUnit object can be constructed with the correct attributes.
            - The id and nameplate_capacity are set as expected.
            - The monthly_expected_generation property matches the input array.
        """
        net_hourly_capacity, monthly_expected_generation, _ = create_test_data()
        unit = HydroUnit(
            id=1,
            nameplate_capacity=100,
            monthly_expected_generation=monthly_expected_generation
        )

        # Assertions
        self.assertEqual(unit.id, 1, "ID does not match")
        self.assertEqual(unit.nameplate_capacity, 100, "Nameplate capacity does not match")
        self.assertTrue(np.array_equal(unit.monthly_expected_generation, monthly_expected_generation))

    def test_proportional_dispatch(self):
        """
        Test proportional hourly dispatch logic in HydroUnit.

        Ensures:
            - Total dispatched energy equals the sum of the expected monthly generation.
            - Within each month, the hourly dispatched energy is proportional to net unmet demand.
            - Dispatched energy per hour does not exceed expectations, and proportional allocation is followed.
            - Provides debug info if mismatches are found for any month.
        """
        net_hourly_capacity, monthly_expected_generation, _ = create_test_data()

        unit = HydroUnit(
            id=1,
            nameplate_capacity=100,
            monthly_expected_generation=monthly_expected_generation
        )

        hourly_capacity = unit._get_hourly_capacity(net_hourly_capacity)

        total_dispatch = hourly_capacity.sum().item()
        expected_dispatch = monthly_expected_generation.sum().item()

        np.testing.assert_allclose(
            total_dispatch, expected_dispatch, atol=1e-5,
            err_msg="Total dispatch does not match expected generation."
        )

        for month in range(1, 13):
            monthly_data = net_hourly_capacity.sel(time=net_hourly_capacity['time.month'] == month)
            proportions = monthly_data / monthly_data.sum()
            expected_monthly_dispatch = monthly_expected_generation.sel(month=month).item()

            proportional_dispatch = (proportions * expected_monthly_dispatch).fillna(0).values

            # Logically check for mismatches
            actual = hourly_capacity.sel(time=hourly_capacity['time.month'] == month).values
            np.testing.assert_allclose(
                actual,
                proportional_dispatch,
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"Dispatch is not proportional for month {month}."
            )

            # Additional detailed diagnostic logging for debugging
            mismatches = np.abs(actual - proportional_dispatch) > (1e-2 + 1e-2 * np.abs(proportional_dispatch))
            if np.any(mismatches):
                print(f"Debug info for month {month}:")
                print("Mismatches found at indices:", np.where(mismatches))
                print("Actual:", actual[mismatches])
                print("Expected:", proportional_dispatch[mismatches])

    def test_dispatch_with_forced_outage(self):
        """
        Test effect of forced outages on hydro dispatch.

        Ensures:
            - When a forced outage rate is applied, the resulting dispatch is not identical to the
              non-outaged version.
            - Asserts at least some hours are affected (i.e., dispatch is reduced or zeroed).
        """
        net_hourly_capacity, monthly_expected_generation, hourly_forced_outage_rate = create_test_data()
        
        unit = HydroUnit(
            id=1,
            nameplate_capacity=100,
            monthly_expected_generation=monthly_expected_generation,
            hourly_forced_outage_rate=hourly_forced_outage_rate
        )

        hourly_capacity = HydroUnit._get_hourly_capacity(unit, net_hourly_capacity)
        
        outage_mask = np.random.random(hourly_capacity.shape) > hourly_forced_outage_rate.values
        adjusted_capacity = hourly_capacity * outage_mask

        #print(hourly_capacity, adjusted_capacity)

        self.assertFalse(np.all(adjusted_capacity.values == hourly_capacity.values), "Forced outage has no effect.")

    def test_probabilistic_capacity(self):
        """
        Test probabilistic dispatch matrix for hydro units.

        Ensures:
            - The returned matrix of dispatched energy has the expected dimensions, including 'trial'.
            - The number of trials in the output matches the number of input scenarios (10 by default).
            - Each trial/scenario runs independently.
        """
        net_hourly_capacity, monthly_expected_generation, hourly_forced_outage_rate = create_test_data()
        units = [HydroUnit(
                    id=1,
                    nameplate_capacity=100,
                    monthly_expected_generation=monthly_expected_generation,
                    hourly_forced_outage_rate=hourly_forced_outage_rate
                 )]

        unit_ds = HydroUnit.to_unit_dataset(units)
        net_hourly_capacity_matrix = xr.concat([net_hourly_capacity] * 10, dim="trial")

        adjusted_matrix = HydroUnit.get_probabilistic_capacity_matrix(unit_ds, net_hourly_capacity_matrix)
        self.assertEqual(adjusted_matrix.dims, net_hourly_capacity_matrix.dims, 
                         "Adjusted matrix dimensions do not match.")
        self.assertEqual(len(adjusted_matrix['trial']), 10, "Unexpected number of trials.")

    def test_hydro_list_to_dataset(self):
        """Hydro unit ids are stored as the energy_unit coordinate."""
        unit = HydroUnit(id=7, nameplate_capacity=3,
                         monthly_expected_generation=xr.DataArray([3.0] * 12, coords=dict(month=list(range(1, 13)))))
        dataset = HydroUnit.to_unit_dataset([unit])
        self.assertIn("energy_unit", dataset.coords)
        self.assertEqual([7], [int(i) for i in dataset.energy_unit.values])

    def test_hydro_dataset_to_list(self):
        """Hydro unit ids survive from_unit_dataset and from_energy_system."""
        from assetra.units import DemandUnit
        from assetra.system import EnergySystemBuilder
        unit = HydroUnit(id=7, nameplate_capacity=3,
                         monthly_expected_generation=xr.DataArray([3.0] * 12, coords=dict(month=list(range(1, 13)))))
        self.assertEqual(7, HydroUnit.from_unit_dataset(HydroUnit.to_unit_dataset([unit]))[0].id)
        b1 = EnergySystemBuilder()
        b1.add_unit(DemandUnit(id=0, hourly_demand=get_sample_time_series([1, 1, 1])))
        b1.add_unit(unit)
        b2 = EnergySystemBuilder.from_energy_system(b1.build())  # raised a duplicate-id RuntimeWarning before the fix
        self.assertIn(7, [u.id for u in b2.energy_units])

if __name__ == "__main__":
    unittest.main()
