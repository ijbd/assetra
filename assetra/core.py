'''TODO: Overview of different classes and internal model.

Examples:

    TODO: link to usage page

Attributes:

    TODO
'''
from __future__ import annotations
from abc import abstractmethod, ABC
from logging import getLogger
from dataclasses import dataclass
from pathlib import Path

# external
import numpy as np
import xarray as xr

log = getLogger(__name__)


# ENERGY UNIT(S)


@dataclass(frozen=True)
class EnergyUnit(ABC):
    """Abstract base class for all energy units.

    Energy units are the fundamental building blocks of energy systems in
    the assetra model. This base class defines an interface which allows the
    assetra model to save/load pre-existing energy systems from files and run
    probabilistic simulations with unique energy unit types.

    Args:
        id (int): Unique identifying number, used to ensure energy units are
            not added more than once to an energy system
        nameplate_capacity (float) : Nameplate capacity of the energy unit in
            units of power (to be kept consistend between units). For some
            units defining the nameplate capacity may not make physical sense,
            e.g. demand units, in which case the nameplate capacity should be
            set to zero.
    """

    # TODO add example for custom unit
    id: int
    nameplate_capacity: float

    @staticmethod
    @abstractmethod
    def to_unit_dataset(units: list[EnergyUnit]) -> xr.Dataset:
        """Convert a list of energy units of the derived class type into an
        xarray dataset.

        For different energy units, different dataset
        dimensions and coordinates may be appropriate.

        Args:
            units (list[EnergyUnit]): List of of energy units of the derived
                class type

        Returns:
            xr.Dataset: Dataset storing sufficient information to (1) fully
                reconstruct the list of energy units from which it is created
                and (2) generate hourly capacity time series with the
                EnergyUnit.get_probabilistic_capacity_matrix function
        """
        pass

    @staticmethod
    @abstractmethod
    def from_unit_dataset(unit_dataset: xr.Dataset) -> list[EnergyUnit]:
        """Convert a unit dataset to a list of energy units of the derived
        energy unit type.

        This is the inverse to the derived EnergyUnit.to_unit_dataset function

        Args:
            unit_dataset (xr.Dataset): Unit dataset with structure and content
                defined in the derived EnergyUnit.to_unit_dataset function

        Returns:
            list[EnergyUnit]: List of energy units of the derived class type
        """

    @staticmethod
    @abstractmethod
    def get_probabilistic_capacity_matrix(
        unit_dataset: xr.Dataset, net_hourly_capacity_matrix: xr.DataArray
    ) -> xr.DataArray:
        """Return probabilistic hourly capacity matrix for a fleet of energy
        units of the derived energy unit type.

        Take the unit dataset and create a matrix representing the total hourly
        capacity of all energy units for some number of monte carlo trials. The
        hours and number of trials should match the net hourly capacity matrix.

        Args:
            unit_dataset (xr.Dataset): Unit dataset for the derived energy unit
                type, e.g. generated with the derived
                EnergyUnit.to_unit_dataset function
            net_hourly_capacity_matrix (xr.DataArray): Probabilistic net hourly
                capacity matrix with dimensions (trials, time) and shape
                (# of trials, # of hours)

        Returns:
            xr.DataArray: Combined hourly capacity for all units in the unit
                dataset for a determined number of Monte Carlo trials. The
                dimensions and coordinates of this matrix should match the net
                hourly capacity matrix
        """
        pass


@dataclass(frozen=True)
class StaticUnit(EnergyUnit):
    """Derived energy unit class.

    A static energy unit is neither stochastic nor volatile. A single
    hourly capacity profile is used in all probabilistic capacity trials.
    For example, a historical demand profile be fully accounted for in all
    trials of a probabilistic simulation.

    Args:
        id (int): Unique identifying number
        nameplate_capacity (float) : Nameplate capacity of the energy unit in
            units of power
        hourly_capacity (xr.DataArray) : Hourly capacity contained in DataArray
            with dimension (time) and datetime coordinates.
    """

    hourly_capacity: xr.DataArray

    @staticmethod
    def to_unit_dataset(units: list[StaticUnit]) -> xr.Dataset:
        """Convert a list of static energy units into an xarray dataset

        Args:
            units (list[StaticUnit]): List of of static energy units

        Returns:
            xr.Dataset: Dataset with dimensions (energy_unit, time) and
                variables (nameplate_capacity[energy_unit],
                hourly_capacity[energy_unit, time]). Coordinates for the
                energy_unit and time dimensions are energy unit IDs and
                hourly datetime indices, respectively.
        """
        # TODO check consistent timeframes or auto-fill zeros
        # build dataset
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(
                    ["energy_unit"],
                    [unit.nameplate_capacity for unit in units],
                ),
                hourly_capacity=(
                    ["energy_unit", "time"],
                    [unit.hourly_capacity for unit in units],
                ),
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units],
                time=units[0].hourly_capacity.time if len(units) > 0 else [],
            ),
        )

        return unit_dataset

    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset) -> list[StaticUnit]:
        """Convert a static unit dataset to a list of static energy units.

        This is the inverse StaticUnit.to_unit_dataset function

        Args:
            unit_dataset (xr.Dataset): Unit dataset with structure and content
                defined in the derived StaticUnit.to_unit_dataset function

        Returns:
            list[StaticUnit]: List of static energy units
        """
        # build list
        units = []

        for id in unit_dataset.energy_unit:
            units.append(
                StaticUnit(
                    int(id),
                    int(unit_dataset.nameplate_capacity.loc[id]),
                    unit_dataset.hourly_capacity.loc[id],
                )
            )

        return units

    @staticmethod
    def get_probabilistic_capacity_matrix(
        unit_dataset: xr.Dataset, net_hourly_capacity_matrix: xr.DataArray
    ) -> xr.DataArray:
        """Return probabilistic hourly capacity matrix for a static unit
        dataset.

        For static units, combine hourly capacity profiles for all energy units
        in the unit dataset and broadcast the result across all trials

        Args:
            unit_dataset (xr.Dataset): Static unit dataset, as generated by
                StaticUnit.to_unit_dataset function
            net_hourly_capacity_matrix (xr.DataArray): Probabilistic net hourly
                capacity matrix with dimensions (trials, time) and shape
                (# of trials, # of hours)

        Returns:
            xr.DataArray: Combined hourly capacity for all units in the unit
                dataset with the same dimensions and shape as net hourly
                capacity matrix
        """
        # time-indexing
        unit_dataset = unit_dataset.sel(time=net_hourly_capacity_matrix.time)

        # sum across capacity units
        probabilistic_capacity_matrix = unit_dataset["hourly_capacity"].sum(
            dim="energy_unit"
        )

        # to xarray
        probabilistic_capacity_matrix = (
            xr.zeros_like(net_hourly_capacity_matrix)
            + probabilistic_capacity_matrix
        )

        return probabilistic_capacity_matrix


@dataclass(frozen=True)
class StochasticUnit(EnergyUnit):
    """Derived energy unit class.

    A stochastic energy unit uses time-varying forced outage rates to sample
    indepenedent outages throughout the simulation period. Stochastic units are
    non-volatile, meaning that while hourly capacity profiles vary between
    trials in a probabilistic simulation, the profiles do not depend on system
    conditions and only need to be sampled once

    Args:
        id (int): Unique identifying number
        nameplate_capacity (float) : Nameplate capacity of the energy unit in
            units of power
        hourly_capacity (xr.DataArray) : Hourly capacity contained in DataArray
            with dimension (time) and datetime coordinates
        hourly_forced_outage_rate (xr.DataArray) : Hourly forced outage rate
            as decimal percents (e.g. 5% -> 0.05) contained in DataArray with
            dimension (time) and datetime coordinates. Should be a parallel
            matrix to hourly_capacity
    """

    hourly_capacity: xr.DataArray
    hourly_forced_outage_rate: xr.DataArray

    @staticmethod
    def to_unit_dataset(units: list[StochasticUnit]):
        """Convert a list of stochastic energy units into an xarray dataset

        Args:
            units (list[StochasticUnit]): List of of stochastic energy units

        Returns:
            xr.Dataset: Dataset with dimensions (energy_unit, time) and
                variables (nameplate_capacity[energy_unit],
                hourly_capacity[energy_unit, time]),
                hourly_forced_outage_rate[energy_unit, time]. Coordinates for
                the energy_unit and time dimensions are energy unit IDs and
                hourly datetime indices, respectively.
        """
        # TODO check consistent timeframes or auto-fill zeros
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(
                    ["energy_unit"],
                    [unit.nameplate_capacity for unit in units],
                ),
                hourly_capacity=(
                    ["energy_unit", "time"],
                    [unit.hourly_capacity for unit in units],
                ),
                hourly_forced_outage_rate=(
                    ["energy_unit", "time"],
                    [unit.hourly_forced_outage_rate for unit in units],
                ),
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units],
                time=units[0].hourly_capacity.time if len(units) > 0 else [],
            ),
        )
        return unit_dataset

    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset) -> list[StochasticUnit]:
        """Convert a stochastic unit dataset to a list of stochastic energy units.

        This is the inverse to StochasticUnit.to_unit_dataset function

        Args:
            unit_dataset (xr.Dataset): Unit dataset with structure and content
                defined in the derived StochasticUnit.to_unit_dataset function

        Returns:
            list[StochasticUnit]: List of stochastic units
        """
        # build list
        units = []

        for id in unit_dataset.energy_unit:
            units.append(
                StochasticUnit(
                    id,
                    unit_dataset.nameplate_capacity.loc[id],
                    unit_dataset.hourly_capacity.loc[id],
                    unit_dataset.hourly_forced_outage_rate.loc[id],
                )
            )

        return units

    @staticmethod
    def get_probabilistic_capacity_matrix(
        unit_dataset: xr.Dataset, net_hourly_capacity_matrix: xr.DataArray
    ) -> xr.DataArray:
        """Return probabilistic hourly capacity matrix for a stochastic unit
        dataset.

        For stochastic units, sample hourly independent outages in for units
        in all trials. Outages are sampled hourly for every unit and trial.
        Random numbers are drawn from the range 0 to 1, and where samples are
        less than the hourly forced outage rate, the effective capacity of
        that energy unit in that hour and trial is set to 0. The probabilistic
        capacity matrix is the aggregation of sampled capacities across energy
        units

        Args:
            unit_dataset (xr.Dataset): Static unit dataset, as generated by
                StaticUnit.to_unit_dataset function
            net_hourly_capacity_matrix (xr.DataArray): Probabilistic net hourly
                capacity matrix with dimensions (trials, time) and shape
                (# of trials, # of hours)

        Returns:
            xr.DataArray: Combined hourly capacity for all units in the unit
                dataset with the same dimensions and shape as net hourly
                capacity matrix
        """
        # time-indexing
        unit_dataset = unit_dataset.sel(time=net_hourly_capacity_matrix.time)

        # sample outages
        probabilistic_capacity_matrix = np.where(
            np.random.random_sample(
                (
                    net_hourly_capacity_matrix.sizes["trial"],
                    unit_dataset.sizes["energy_unit"],
                    unit_dataset.sizes["time"],
                )
            )
            > unit_dataset["hourly_forced_outage_rate"].values,
            unit_dataset["hourly_capacity"].values,
            0,
        ).sum(axis=1)

        # to xarray
        probabilistic_capacity_matrix = net_hourly_capacity_matrix.copy(
            data=probabilistic_capacity_matrix
        )

        return probabilistic_capacity_matrix


@dataclass(frozen=True)
class StorageUnit(EnergyUnit):
    """Derived energy unit class.

    A storage unit is a state-dependent, volatile energy unit. The available
    capacity of a storage unit depends on its state of charge and on the needs
    of the system. As opposed to static and stochastic units, which require
    hourly time series, storage unit operation is characterized by a handful of
    scalar parameters

    Args:
        id (int): Unique identifying number
        nameplate_capacity (float) : Nameplate capacity in units of power. For
            storage, typically the discharge rate
        charge_rate (float) : Charge rate in units of power
        discharge_rate (float) : Discharge rate in units of power
        charge_capacity (float) : Maximum charge capacity in units of energy
        roundtrip_efficiency (float) : Roundtrip efficiency as decimal percent
    """

    charge_rate: float
    discharge_rate: float
    charge_capacity: float
    roundtrip_efficiency: float

    def _get_hourly_capacity(
        charge_rate: float,
        discharge_rate: float,
        charge_capacity: float,
        roundtrip_efficiency: float,
        net_hourly_capacity: xr.DataArray,
    ) -> xr.DataArray:
        """Greedy storage dispatch

        Args:
            charge_rate (float) : Charge rate in units of power
            discharge_rate (float) : Discharge rate in units of power
            charge_capacity (float) : Maximum charge capacity in units of
                energy
            roundtrip_efficiency (float) : Roundtrip efficiency as decimal
                percent
            net_hourly_capacity (xr.DataArray): Net capacity contained in
                DataArray with dimension (time) and hourly datetime
                coordinates

        Returns:
            xr.DataArray: Hourly capacity contained in DataArray with same
                shape as net hourly capacity
        """
        # TODO skip irrelevant days for average-case speed-up?
        # initialize full storage unit
        efficiency = roundtrip_efficiency**0.5

        def charge_storage(excess_capacity: float, current_charge: float):
            capacity = -min(
                charge_rate,
                (charge_capacity - current_charge) / efficiency,
                excess_capacity,
            )
            current_charge -= capacity * efficiency

            return capacity, current_charge

        def discharge_storage(unmet_demand: float, current_charge: float):
            capacity = min(
                discharge_rate / efficiency,
                current_charge,
                unmet_demand / efficiency,
            )
            current_charge -= capacity

            return capacity * efficiency, current_charge

        def dispatch_storage(net_hourly_capacity: float):
            # initialize storage unit as full
            current_charge = float(charge_capacity)

            for net_capacity in net_hourly_capacity.values:
                capacity = 0
                if net_capacity < 0:
                    if current_charge > 0:
                        # unmet demand and avaiable charge
                        capacity, current_charge = discharge_storage(
                            -net_capacity, current_charge
                        )
                elif current_charge < charge_capacity:
                    # excess capacity and not fully charged
                    capacity, current_charge = charge_storage(
                        net_capacity, current_charge
                    )
                yield capacity

        # simulate dispatch
        hourly_capacity = net_hourly_capacity.copy(
            data=[
                capacity for capacity in dispatch_storage(net_hourly_capacity)
            ]
        )

        return hourly_capacity

    @staticmethod
    def to_unit_dataset(units: list[StorageUnit]) -> xr.Dataset:
        """Convert a list of storage units into an xarray dataset

        Args:
            units (list[StorageUnit]): List of of storage energy units

        Returns:
            xr.Dataset: Dataset with dimensions (energy_unit) and
                variables (nameplate_capacity[energy_unit],
                charge_rate[energy_unit], discharge_rate[energy_unit],
                charge_capacity[energy_unit], roundtrip_efficiency[energy_unit]
                hourly_forced_outage_rate[energy_unit, time]). Coordinates for
                the energy_unit dimension are energy unit IDs
        """
        # build dataset
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(
                    ["energy_unit"],
                    [unit.nameplate_capacity for unit in units],
                ),
                charge_rate=(
                    ["energy_unit"],
                    [unit.charge_rate for unit in units],
                ),
                discharge_rate=(
                    ["energy_unit"],
                    [unit.discharge_rate for unit in units],
                ),
                charge_capacity=(
                    ["energy_unit"],
                    [unit.charge_capacity for unit in units],
                ),
                roundtrip_efficiency=(
                    ["energy_unit"],
                    [unit.roundtrip_efficiency for unit in units],
                ),
            ),
            coords=dict(energy_unit=[unit.id for unit in units]),
        )

        return unit_dataset

    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset) -> list[StorageUnit]:
        """Convert a storage unit dataset to a list of storage units.

        This is the inverse StorageUnit.to_unit_dataset function

        Args:
            unit_dataset (xr.Dataset): Unit dataset with structure and content
                defined in the derived StorageUnit.to_unit_dataset function

        Returns:
            list[StorageUnit]: List of storage units
        """
        # build list
        units = []

        for id, nc, cr, dr, cc, re in zip(
            unit_dataset.energy_unit,
            unit_dataset.nameplate_capacity,
            unit_dataset.charge_rate,
            unit_dataset.discharge_rate,
            unit_dataset.charge_capacity,
            unit_dataset.roundtrip_efficiency,
        ):
            units.append(
                StorageUnit(
                    id=int(id),
                    nameplate_capacity=float(nc),
                    charge_rate=float(cr),
                    discharge_rate=float(dr),
                    charge_capacity=float(cc),
                    roundtrip_efficiency=float(re),
                )
            )

        return units

    @staticmethod
    def get_probabilistic_capacity_matrix(
        unit_dataset: xr.Dataset, net_hourly_capacity_matrix: xr.DataArray
    ) -> xr.DataArray:
        """Return probabilistic hourly capacity matrix for a storage unit
        dataset.

        For storage units, it is necessary to dispatch units every hour and
        iteration sequentially. The dispatch policy implemented in
        StorageUnit._get_hourly_capacity is a greedy policy to minimize
        expected unserved energy. Units are dispatched according to the order
        they appear in the unit dataset

        Args:
            unit_dataset (xr.Dataset): Storage unit dataset, as generated by
                StorageUnit.to_unit_dataset function
            net_hourly_capacity_matrix (xr.DataArray): Probabilistic net hourly
                capacity matrix with dimensions (trials, time) and shape
                (# of trials, # of hours)

        Returns:
            xr.DataArray: Combined hourly capacity for all units in the unit
                dataset with the same dimensions and shape as net hourly
                capacity matrix
        """
        units = StorageUnit.from_unit_dataset(unit_dataset)

        net_adj_hourly_capacity_matrix = net_hourly_capacity_matrix.copy()
        for unit in units:
            for trial in net_adj_hourly_capacity_matrix:
                trial += StorageUnit._get_hourly_capacity(
                    unit.charge_rate,
                    unit.discharge_rate,
                    unit.charge_capacity,
                    unit.roundtrip_efficiency,
                    trial,
                )

        return net_adj_hourly_capacity_matrix - net_hourly_capacity_matrix


# for successive simulations (e.g. ELCC), need to differentiate between
# volatile and non-volatile units.
#
# these lists also serve to track all "valid" units that can be added
# to an energy system
NONVOLATILE_UNIT_TYPES = [StaticUnit, StochasticUnit]
VOLATILE_UNIT_TYPES = [StorageUnit]


class EnergySystem:
    """Class responsible for managing unit datasets (built energy systems)

    Args:
        unit_datasets (dict[Type : xr.Dataset]) : Mapping from derived energy
            unit type to its associated unit dataset
    """

    def __init__(self, unit_datasets: dict[type : xr.DataArray] = {}):
        self._unit_datasets = unit_datasets

        # check input
        for key in unit_datasets:
            if key not in (VOLATILE_UNIT_TYPES + NONVOLATILE_UNIT_TYPES):
                log.error(
                    "Constructing energy system with invalid unit dataset"
                )
                raise RuntimeWarning

    @property
    def nameplate_capacity(self) -> float:
        """Sum of nameplate capacities for all energy units in a system

        Returns:
            float : Total nameplate capacity of a system
        """
        return sum(
            d["nameplate_capacity"].sum() for d in self._unit_datasets.values()
        )

    @property
    def unit_datasets(self) -> dict[type : xr.Dataset]:
        """Access underlying unit_datasets. This is not a deep copy.

        Returns:
            dict[type : xr.Dataset] : Mapping from derived energy
                unit type to its associated unit dataset
        """
        return self._unit_datasets

    def get_system_by_type(self, unit_type: list[type] | type) -> EnergySystem:
        """Return a system comprised of the subset of unit datasets
        corresponding to one (or more) energy unit types.

        For example, get a [sub]system with only the volatile or non-volatile
        units of a system

        Args:
            unit_type (list[type] | type): Either a derived energy unit type or
                a list of derived energy unit types.

        Returns:
            EnergySystem: An energy system whose unit datasets are a sub-set of
                of this energy system.
        """
        if isinstance(unit_type, (list, tuple)):
            unit_datasets = {
                ut: ud
                for ut, ud in self._unit_datasets.items()
                if ut in unit_type
            }
            return EnergySystem(unit_datasets)
        elif unit_type in VOLATILE_UNIT_TYPES + NONVOLATILE_UNIT_TYPES:
            return EnergySystem({unit_type: self._unit_datasets[unit_type]})

    def save(self, directory: Path) -> None:
        """Save energy system to a directory. Unit datasets are saved as netcdf
        files

        Args:
            directory (Path): Path to which energy system is saved. This path
                should either be empty or not exist yet.
        """
        # TODO check for non-empty directory
        # TODO create directory
        for unit_type, dataset in self._unit_datasets.items():
            dataset_file = Path(directory, unit_type.__name__ + ".assetra.nc")
            dataset.to_netcdf(dataset_file)

    def load(self, directory: Path) -> None:
        """Load energy system from a saved directory

        Args:
            directory (Path): Path from which energy system is loaded. This
                should be the same path passed to EnergySystem.save
        """
        # TODO check for existing dataset?
        self._unit_datasets = dict()

        for unit_type in NONVOLATILE_UNIT_TYPES + VOLATILE_UNIT_TYPES:
            dataset_file = Path(directory, unit_type.__name__ + ".assetra.nc")

            if dataset_file.exists():
                self._unit_datasets[unit_type] = xr.open_dataset(dataset_file)


class EnergySystemBuilder:
    """Class responsible for managing energy units and building energy systems.

    Internally, we *try* to think of energy systems as immutable. There is no
    way to directly add, remove, or modify energy units to/from/in an  system.
    The reason for this is to make explicitly clear to users that higher level
    objects do not track the state of lower-level objects. For example, if a
    user wants to modify a system for which a probabilistic simulation has
    already been evaluated, it would be tedious to both recognize the system
    modification from the simulation object and preserve computation from
    the existing evaluation.

    Further, we want to make efficient use of data structures for larger
    simulations. For example, it is both time- and memory- efficient to operate
    on whole fleets of energy units via matrix operation rather than evaluating
    each unit individually. This also offers a pathway to future
    parallelization.

    On the other hand, it is important for users to modify systems, i.e. add or
    remove units at will, and it is convenient to think of energy units as
    individual conceptual objects (not as fleets).

    To summarize, the internal energy system model should be immutable and
    operate on fleets of energy units, while the external model should be
    modifiable and treat energy units as individual objects. The
    EnergySystemBuilder class acts as a bridge between these two models
    """

    def __init__(self):
        self._energy_units = []

    @property
    def energy_units(self) -> tuple[EnergyUnit]:
        """
        Returns:
            tuple[EnergyUnit]: Energy units added to builder object.
        """
        return tuple(self._energy_units)

    @property
    def size(self) -> int:
        """
        Returns:
            int: Number of energy units added to builder object."""
        return len(self._energy_units)

    def add_unit(self, energy_unit: EnergyUnit) -> None:
        """Add an energy unit to the system builder object.

        Args:
            energy_unit (EnergyUnit): Energy unit to add to system builder

        Raises:
            RuntimeWarning: Invalid energy unit type added to system builder
            RuntimeWarning: Duplicate unit added to energy system builder
        """
        # check for valid energy unit
        if (
            type(energy_unit)
            not in NONVOLATILE_UNIT_TYPES + VOLATILE_UNIT_TYPES
        ):
            log.warning("Invalid type added to energy system builder")
            raise RuntimeWarning()

        # check for duplicates
        if energy_unit.id in [u.id for u in self._energy_units]:
            log.warning("Duplicate unit added to energy system builder")
            raise RuntimeWarning()

        # add unit to internal list
        self._energy_units.append(energy_unit)

    def remove_unit(self, energy_unit: EnergyUnit) -> None:
        """Remove an energy unit from the system builder object.

        Args:
            energy_unit (EnergyUnit): Energy unit to remove from system builder
        """
        try:
            self._energy_units.remove(energy_unit)
        except KeyError:
            log.warning("Unit to remove not found in energy system builder")

    def build(self) -> EnergySystem:
        """Return a populated EnergySystem instance. Take energy units added to
        the builder object, compile each unit type into a unit dataset (fleet),
        and instantiate an EnergySystem with the resulting unit dataset
        dictionary.

        This is the recommended method to instantiate EnergySystem objects

        Returns:
            EnergySystem: Populated energy system instance
        """
        unit_datasets = dict()

        # populate unit datasets
        for unit_type in NONVOLATILE_UNIT_TYPES + VOLATILE_UNIT_TYPES:
            # get unit by type
            units = [
                unit for unit in self.energy_units if type(unit) is unit_type
            ]

            # get unit dataset
            if len(units) > 0:
                unit_datasets[unit_type] = unit_type.to_unit_dataset(units)

        return EnergySystem(unit_datasets)

    @staticmethod
    def from_energy_system(energy_system: EnergySystem) -> EnergySystemBuilder:
        """Return a populated EnergySystemBuilder instance. Take energy unit
        datasets from an energy system, convert datasets into
        individual energy units, and add units to a new EnergySystemBuilder
        object

        This is the inverse to EnergySystem.build, and is useful for
        modifying energy systems which have been loaded directly from file
        using the EnergySystem.load function

        Args:
            energy_system (EnergySystem): Populated energy system

        Returns:
            EnergySystemBuilder: Populated builder instance
        """
        builder = EnergySystemBuilder()

        for unit_type, unit_dataset in energy_system.unit_datasets.items():
            units = unit_type.from_unit_dataset(unit_dataset)
            for unit in units:
                builder.add_unit(unit)

        return builder
