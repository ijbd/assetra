from __future__ import annotations
from logging import getLogger
from pathlib import Path

# external
import xarray as xr

# package
from assetra.units import EnergyUnit, RESPONSIVE_UNIT_TYPES, NONRESPONSIVE_UNIT_TYPES

LOG = getLogger(__name__)

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
            if key not in (RESPONSIVE_UNIT_TYPES + NONRESPONSIVE_UNIT_TYPES):
                LOG.error(
                    "Constructing energy system with invalid unit dataset"
                )
                raise RuntimeWarning

    @property
    def system_capacity(self) -> float:
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

        For example, get a [sub]system with only the responsive or non-responsive
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
        elif unit_type in RESPONSIVE_UNIT_TYPES + NONRESPONSIVE_UNIT_TYPES:
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

        for unit_type in NONRESPONSIVE_UNIT_TYPES + RESPONSIVE_UNIT_TYPES:
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
            not in NONRESPONSIVE_UNIT_TYPES + RESPONSIVE_UNIT_TYPES
        ):
            LOG.warning("Invalid type added to energy system builder")
            raise RuntimeWarning()

        # check for duplicates
        if energy_unit.id in [u.id for u in self._energy_units]:
            LOG.warning("Duplicate unit added to energy system builder")
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
            LOG.warning("Unit to remove not found in energy system builder")

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
        for unit_type in NONRESPONSIVE_UNIT_TYPES + RESPONSIVE_UNIT_TYPES:
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
