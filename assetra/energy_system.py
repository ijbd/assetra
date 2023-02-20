from __future__ import annotations
from abc import abstractmethod, ABC
from logging import getLogger
from datetime import datetime
from dataclasses import dataclass
from typing import List

# external
import numpy as np
import xarray as xr

log = getLogger(__name__)

# ENERGY UNIT(S)


@dataclass(frozen=True)
class EnergyUnit(ABC):
    nameplate_capacity: float

    @staticmethod
    @abstractmethod
    def to_unit_dataset(units: List[EnergyUnit]):
        pass

    @staticmethod
    @abstractmethod
    def from_unit_dataset(dataset: xr.Dataset):
        pass


@dataclass(frozen=True)
class StaticUnit(EnergyUnit):
    hourly_capacity: xr.DataArray

    @staticmethod
    def to_unit_dataset(units: List[StaticUnit]):
        # check for static units
        if len(units) == 0:
            log.info('No static units available. Skipping build.')
            return None

        # build dataset
        unit_dataset = xr.DataSet(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [unit.nameplate_capacity for unit in units]),
                hourly_capacity=(['energy_unit', 'time'], [unit.hourly_capacity for unit in units])
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units],
                time=units[0].time if len(units) > 0 else []
            )
        )

        return unit_dataset


@dataclass(frozen=True)
class StochasticUnit(EnergyUnit):
    hourly_capacity: xr.DataArray
    hourly_forced_outage_rate: xr.DataArray

    @staticmethod
    def to_unit_dataset(units: List[StochasticUnit]):
        unit_dataset = xr.DataSet(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [unit.nameplate_capacity for unit in units]),
                hourly_capacity=(['energy_unit', 'time'], [unit.hourly_capacity for unit in units]),
                hourly_forced_outage_rate=(['energy_unit', 'time'], [unit.hourly_forced_outage_rate for unit in units])
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units],
                time=units[0].time if len(units) > 0 else []
            )
        )

        return unit_dataset


@dataclass(frozen=True)
class SequentialUnit(EnergyUnit):

    @abstractmethod
    def get_hourly_capacity(self, net_hourly_capacity):
        pass


@dataclass(frozen=True)
class StorageUnit(SequentialUnit):
    charge_rate: float
    discharge_rate: float
    duration: float
    roundtrip_efficiency: float

    def __post_init__(self):
        self.charge_capacity = self.discharge_rate * self.duration
        self.efficiency = self.roundtrip_efficiency**0.5

    def get_hourly_capacity(
        self,
        net_hourly_capacity: xr.DataArray
    ):
        # initialize full storage unit
        current_charge = self.charge_capacity
        hourly_capacity = xr.zeros_like(net_hourly_capacity)

        # simulate dispatch
        for i, net_capacity in enumerate(net_hourly_capacity):
            hourly_capacity[i], current_charge = self._dispatch_storage(
                net_capacity, current_charge
            )

        return hourly_capacity

    def _dispatch_storage(self, net_capacity: float, current_charge: float):
        capacity = 0
        if net_capacity < 0:
            # unmet demand
            if current_charge > 0:
                capacity, current_charge = self._discharge_storage(
                    -net_capacity, current_charge
                )
        else:
            # excess capacity
            if current_charge < self._charge_capacity:
                capacity, current_charge = self._charge_storage(
                    net_capacity, current_charge
                )

        return capacity, current_charge

    def _charge_storage(self, excess_capacity: float, current_charge: float):
        capacity = -min(
            self._charge_rate,
            (self._charge_capacity - current_charge) / self._efficiency,
            excess_capacity,
        )
        current_charge -= capacity * self._efficiency

        return capacity, current_charge

    def _discharge_storage(self, unmet_demand: float, current_charge: float):
        capacity = min(
            self._discharge_rate / self._efficiency,
            current_charge,
            unmet_demand / self._efficiency,
        )
        current_charge -= capacity

        return capacity * self._efficiency, current_charge

    @staticmethod
    def to_unit_dataset(units: List[StaticUnit]):
        # check for storage units
        if len(units) == 0:
            log.info('No storage units available. Skipping build.')
            return None

        # build dataset
        unit_dataset = xr.DataSet(
            data_vars=dict(
                charge_rate=(['energy_unit'], [unit.charge_rate for unit in units]),
                discharge_rate=(['energy_unit'], [unit.discharge_rate for unit in units]),
                duration=(['energy_unit'], [unit.duration for unit in units]),
                roundtrip_efficiency=(['energy_unit'], [unit.roundtrip_efficiency for unit in units]),
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units]
            )
        )

        return unit_dataset
    

# ENERGY SYSTEM

class EnergySystem:
    """Class responsible for managing energy units."""

    '''IJBD 02-20-2023: WORKING ON ADDING LOGIC FOR BUILDING UNIT DATASETS'''

    def __init__(self):
        self._energy_units = []
        self._modified = False

    @property
    def size(self):
        return len(self._energy_units)

    @property
    def capacity(self):
        return sum([u.nameplate_capacity for u in self._energy_units])

    @property
    def energy_units(self):
        return tuple(self._energy_units)

    @property
    def energy_unit_datasets(self):
        if self._modified:
            self.build()
            return self._energy_unit_datasets

    def add_unit(self, energy_unit: EnergyUnit):
        # TODO check for valid energy unit
        
        # check for duplicates
        if energy_unit.id in [u.id for u in self._energy_units]:
            raise RuntimeError("Duplicate unit placed in energy system.")

        # add unit to internal list
        self._energy_units.append(energy_unit)
        self._modified = True

    def remove_unit(self, energy_unit: EnergyUnit):
        self._energy_units.remove(energy_unit)
        self._modified = True

    def add_system(self, other: EnergySystem):
        for energy_unit in other._energy_units:
            self.add_unit(energy_unit)

    def remove_system(self, other: EnergySystem):
        for energy_unit in other._energy_units:
            self.remove_unit(energy_unit)

    def build(self):

        # update modified flag
        self._modified = False

    def save(self, file):
        pass 

    def load(self, file):
        pass

