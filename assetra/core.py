from __future__ import annotations
from abc import abstractmethod, ABC
from logging import getLogger
from datetime import datetime
from dataclasses import dataclass
from typing import List
from collections import namedtuple

# external
import numpy as np
import xarray as xr

log = getLogger(__name__)

# ENERGY UNIT(S)

@dataclass(frozen=True)
class EnergyUnit(ABC):
    id: int
    nameplate_capacity: float

    @staticmethod
    @abstractmethod
    def to_unit_dataset(units: List[EnergyUnit]):
        pass

    @staticmethod
    @abstractmethod
    def from_unit_dataset(unit_dataset: xr.Dataset):
        pass

    @staticmethod
    @abstractmethod
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        pass


@dataclass(frozen=True)
class StaticUnit(EnergyUnit):
    hourly_capacity: xr.DataArray

    @staticmethod
    def to_unit_dataset(units: List[StaticUnit]):
        # build dataset
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [unit.nameplate_capacity for unit in units]),
                hourly_capacity=(['energy_unit', 'time'], [unit.hourly_capacity for unit in units])
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units],
                time=units[0].hourly_capacity.time if len(units) > 0 else []
            )
        )

        return unit_dataset
    
    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset):
        # TODO
        pass
    
    @staticmethod
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        # time-indexing
        unit_dataset = unit_dataset.sel(time=slice(start_hour, end_hour))

        return unit_dataset.sel(time=slice(start_hour, end_hour))['hourly_capacity'].sum(dim='energy_unit')


@dataclass(frozen=True)
class StochasticUnit(EnergyUnit):
    hourly_capacity: xr.DataArray
    hourly_forced_outage_rate: xr.DataArray

    @staticmethod
    def to_unit_dataset(units: List[StochasticUnit]):
        unit_dataset = xr.Dataset(
            data_vars=dict(
                nameplate_capacity=(['energy_unit'], [unit.nameplate_capacity for unit in units]),
                hourly_capacity=(['energy_unit', 'time'], [unit.hourly_capacity for unit in units]),
                hourly_forced_outage_rate=(['energy_unit', 'time'], [unit.hourly_forced_outage_rate for unit in units])
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units],
                time=units[0].hourly_capacity.time if len(units) > 0 else []
            )
        )
        return unit_dataset
    
    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset):
        # TODO
        pass
    
    @staticmethod
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        # time-indexing
        unit_dataset = unit_dataset.sel(time=slice(start_hour, end_hour))

        # sample outages
        probabilistic_hourly_capacity = np.where(unit_dataset['hourly_capacity'].values, 
            np.random.random_sample((trials, unit_dataset.sizes['energy_unit'], unit_dataset.sizes['time']))
            > unit_dataset['hourly_forced_outage_rate'].values, 0
            ).sum(axis=1)
        
        return probabilistic_hourly_capacity


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

    def get_hourly_capacity(
        self,
        net_hourly_capacity: xr.DataArray
    ):
        # initialize full storage unit
        charge_capacity = self.discharge_rate * self.duration
        efficiency = self.roundtrip_efficiency**0.5
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
    def to_unit_dataset(units: List[StorageUnit]):
        # build dataset
        unit_dataset = xr.Dataset(
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
    
    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset):
        # TODO
        pass
    
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        # TODO
        pass


VALID_UNIT_TYPES = [StaticUnit, StochasticUnit, StorageUnit]


class EnergySystem:
    '''Class responsible for managing energy unit datasets'''
    def __init__(self):
        self.unit_datasets = dict()

    def build(self, energy_units: List[EnergyUnit]):
        for unit_type in VALID_UNIT_TYPES:
            # get unit by type
            units = [unit for unit in energy_units if type(unit) is unit_type]

            # get unit dataset
            if len(units) > 0:
                unit_datasets[unit_type] = unit_type.to_unit_dataset(units)

        return None
    
    def save(self, directory):
        # TODO save datasets to directory
        pass

    def load(self, directory):
        # TODO load datasets from directory
        pass


class EnergySystemBuilder:
    """Class responsible for managing energy units."""
    # TODO add hints to error messages

    def __init__(self):
        self._energy_units = []

    def add_unit(self, energy_unit: EnergyUnit):
        # check for valid energy unit
        if type(energy_unit) not in VALID_UNIT_TYPES:
            raise RuntimeError("Invalid type added to energy system.")
        
        # check for duplicates
        if energy_unit.id in [u.id for u in self._energy_units]:
            raise RuntimeError("Duplicate unit placed in energy system.")

        # add unit to internal list
        self._energy_units.append(energy_unit)

    def remove_unit(self, energy_unit: EnergyUnit):
        self._energy_units.remove(energy_unit)

    def build_system(self):
        system = EnergySystem()
        system.build(self._energy_units)
        return system

    @staticmethod
    def from_energy_system(energy_system):
        # TODO
        pass

