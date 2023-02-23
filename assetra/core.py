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
    def to_unit_dataset(units: List[StaticUnit]) -> xr.Dataset:
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
    def from_unit_dataset(unit_dataset: xr.Dataset) -> List[StaticUnit]:
        # build list
        units = []

        for id in unit_dataset.energy_unit:
            units.append(
                StaticUnit(
                    id,
                    unit_dataset.nameplate_capacity.loc[id],
                    unit_dataset.hourly_capacity.loc[id]
                )
            )

        return units
    
    @staticmethod
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        # time-indexing
        unit_dataset = unit_dataset.sel(time=slice(start_hour, end_hour))

        # sum across capacity units
        probabilistic_capacity_matrix = unit_dataset['hourly_capacity'].sum(dim='energy_unit')

        # to xarray
        probabilistic_capacity_matrix = xr.zeros_like(net_hourly_capacity_matrix) + probabilistic_capacity_matrix

        return probabilistic_capacity_matrix

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
    def from_unit_dataset(unit_dataset: xr.Dataset) -> List[StochasticUnit]:
        # build list
        units = []

        for id in unit_dataset.energy_unit:
            units.append(
                StochasticUnit(
                    id,
                    unit_dataset.nameplate_capacity.loc[id],
                    unit_dataset.hourly_capacity.loc[id],
                    unit_dataset.hourly_forced_outage_rate.loc[id]
                )
            )

        return units
    
    @staticmethod
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        # time-indexing
        unit_dataset = unit_dataset.sel(time=slice(start_hour, end_hour))

        # sample outages
        probabilistic_capacity_matrix = np.where(np.random.random_sample((trials, unit_dataset.sizes['energy_unit'], unit_dataset.sizes['time']))
            > unit_dataset['hourly_forced_outage_rate'].values, unit_dataset['hourly_capacity'].values, 0
            ).sum(axis=1)
        
        # to xarray
        probabilistic_capacity_matrix = xr.zeros_like(net_hourly_capacity_matrix) + probabilistic_capacity_matrix
        
        return probabilistic_capacity_matrix

@dataclass(frozen=True)
class StorageUnit(EnergyUnit):
    charge_rate: float
    discharge_rate: float
    charge_capacity: float
    roundtrip_efficiency: float

    def get_hourly_capacity(
        charge_rate: float,
        discharge_rate: float,
        charge_capacity: float,
        roundtrip_efficiency: float,
        net_hourly_capacity: xr.DataArray
    ):
        # TODO skip irrelevant days for average-case speed-up
        # initialize full storage unit
        efficiency = roundtrip_efficiency**0.5
        current_charge = charge_capacity

        def charge_storage(excess_capacity: float, current_charge: float):
            capacity = -min(
                charge_rate,
                (charge_capacity - current_charge) / efficiency,
                excess_capacity,
            )
            current_charge -= capacity * efficiency

            return capacity
        
        def discharge_storage(unmet_demand: float, current_charge: float):
            capacity = min(
                discharge_rate / efficiency,
                current_charge,
                unmet_demand / efficiency,
            )
            current_charge -= capacity

            return capacity * efficiency


        def dispatch_storage(net_hourly_capacity: float):
            current_charge = charge_capacity

            for net_capacity in net_hourly_capacity:
                capacity = 0
                if net_capacity < 0 and current_charge > 0:
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
            data=[capacity for capacity in dispatch_storage(net_hourly_capacity)]
        )

        return hourly_capacity

    @staticmethod
    def to_unit_dataset(units: List[StorageUnit]):
        # build dataset
        unit_dataset = xr.Dataset(
            data_vars=dict(
                charge_rate=(['energy_unit'], [unit.charge_rate for unit in units]),
                discharge_rate=(['energy_unit'], [unit.discharge_rate for unit in units]),
                charge_capacity=(['energy_unit'], [unit.charge_capacity for unit in units]),
                roundtrip_efficiency=(['energy_unit'], [unit.roundtrip_efficiency for unit in units]),
            ),
            coords=dict(
                energy_unit=[unit.id for unit in units]
            )
        )

        return unit_dataset
    
    @staticmethod
    def from_unit_dataset(unit_dataset: xr.Dataset) -> List[StorageUnit]:
        # build list
        units = []

        for id in unit_dataset.energy_unit:
            units.append(
                StorageUnit(
                    id,
                    unit_dataset.nameplate_capacity.loc[id],
                    unit_dataset.charge_rate.loc[id],
                    unit_dataset.discharge_rate.loc[id],
                    unit_dataset.charge_capacity.loc[id],
                    unit_dataset.roundtrip_efficiency.loc[id]
                )
            )

        return units
    
    @staticmethod
    def get_probabilistic_capacity_matrix(unit_dataset: xr.Dataset, start_hour: datetime, end_hour: datetime, trials: int, net_hourly_capacity_matrix: xr.DataArray):
        net_adj_hourly_capacity_matrix = net_hourly_capacity_matrix.copy()
        for unit in unit_dataset.energy_unit:
            for trial in net_adj_hourly_capacity_matrix:
                trial += StorageUnit.get_hourly_capacity(
                    unit_dataset.charge_rate.loc[unit],
                    unit_dataset.discharge_rate.loc[unit],
                    unit_dataset.charge_capacity.loc[unit],
                    unit_dataset.roundtrip_efficiency.loc[unit],
                    trial
                )
        
        return net_adj_hourly_capacity_matrix - net_hourly_capacity_matrix

VALID_UNIT_TYPES = [StaticUnit, StochasticUnit, StorageUnit]

class EnergySystem:
    '''Class responsible for managing energy unit datasets'''
    def __init__(self, energy_units: List[EnergyUnit]):
        self.unit_datasets = dict()
        
        # populate unit datasets
        for unit_type in VALID_UNIT_TYPES:
            # get unit by type
            units = [unit for unit in energy_units if type(unit) is unit_type]

            # get unit dataset
            if len(units) > 0:
                self.unit_datasets[unit_type] = unit_type.to_unit_dataset(units)
    
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

    def build(self):
        system = EnergySystem(self._energy_units)
        return system

    @staticmethod
    def from_energy_system(energy_system: EnergySystem):
        builder = EnergySystemBuilder()

        for unit_type, unit_dataset in energy_system.unit_datasets.items():
            units = unit_type.from_unit_dataset(unit_dataset)
            for unit in units:
                builder.add_unit(unit)

        return builder
    
