from __future__ import annotations
from abc import abstractmethod, ABC
from logging import getLogger
from datetime import datetime, timedelta

# external
import numpy as np
import xarray as xr

log = getLogger(__name__)

# ENERGY UNIT(S)


class EnergyUnit(ABC):
    def __init__(self, id: int, nameplate_capacity: float):
        self._id = id
        self._nameplate_capacity = nameplate_capacity

    # READ-ONLY VARIABLES

    @property
    def nameplate_capacity(self):
        return self._nameplate_capacity

    @property
    def id(self):
        return self._id

    # METHODS

    @abstractmethod
    def get_hourly_capacity(
        self,
        start_hour: datetime,
        end_hour: datetime,
        net_hourly_capacity: xr.DataArray = None,
    ):
        """Returns a single instance of the hourly capacity of the
        generating unit."""
        pass


class StaticUnit(EnergyUnit):
    """Class responsible for returning capacity profile of non-stochastic units
    (i.e. system loads)."""

    def __init__(
        self, id: int, nameplate_capacity: float, hourly_capacity: xr.DataArray
    ):
        EnergyUnit.__init__(self, id, nameplate_capacity)
        self._hourly_capacity = hourly_capacity

    def get_hourly_capacity(
        self,
        start_hour: datetime,
        end_hour: datetime,
        net_hourly_capacity: xr.DataArray = None,
    ):
        return self._hourly_capacity.sel(time=slice(start_hour, end_hour))


class DemandUnit(StaticUnit):
    """Class responsible for returning capacity profile of fixed demand units
    (i.e. system loads)."""

    def __init__(self, id: int, hourly_demand: xr.DataArray):
        StaticUnit.__init__(
            self, id, nameplate_capacity=0, hourly_capacity=-hourly_demand
        )


class ConstantDemandUnit(EnergyUnit):
    def __init__(self, id: int, demand: float):
        EnergyUnit.__init__(self, id, nameplate_capacity=0)
        self._capacity = -demand

    @property
    def demand(self):
        return -self._capacity

    @demand.setter
    def demand(self, new_demand):
        self._capacity = -float(new_demand)

    def get_hourly_capacity(
        self,
        start_hour: datetime,
        end_hour: datetime,
        net_hourly_capacity: xr.DataArray = None,
    ):
        time_stamps = time = xr.date_range(
            start=start_hour, end=end_hour, freq="H", inclusive="both"
        )
        return (
            xr.DataArray(data=np.ones(len(time_stamps)), coords=dict(time=time_stamps))
            * self._capacity
        )


class StochasticUnit(EnergyUnit):
    """Class responsible for returning capacity profile of
    stochastically-sampled units (i.e. generators)."""

    def __init__(
        self,
        id: int,
        nameplate_capacity: float,
        hourly_capacity: xr.DataArray,
        hourly_forced_outage_rate: xr.DataArray,
    ):
        # initialize base class variables
        EnergyUnit.__init__(self, id, nameplate_capacity)
        # initialize stochastic specific variables
        self._hourly_capacity = hourly_capacity
        self._hourly_forced_outage_rate = hourly_forced_outage_rate

    def get_hourly_capacity(
        self,
        start_hour: datetime,
        end_hour: datetime,
        net_hourly_capacity: xr.DataArray = None,
    ):
        # slice time-series
        hourly_capacity_slice = self._hourly_capacity.sel(
            time=slice(start_hour, end_hour)
        )
        hourly_forced_outage_rate_slice = self._hourly_forced_outage_rate.sel(
            time=slice(start_hour, end_hour)
        )

        # draw random samples
        hourly_outage_samples = np.random.random_sample(
            len(hourly_forced_outage_rate_slice)
        )
        return hourly_capacity_slice.where(
            hourly_outage_samples > hourly_forced_outage_rate_slice, 0
        )


class StorageUnit(EnergyUnit):
    """Class responsible for returning capacity profile of state-limited
    storage units"""

    def __init__(
        self,
        id: int,
        charge_rate: float,
        discharge_rate: float,
        duration: float,
        roundtrip_efficiency: float,
    ):
        EnergyUnit.__init__(
            self,
            id,
            nameplate_capacity=discharge_rate,
        )
        self._charge_rate = charge_rate
        self._discharge_rate = discharge_rate
        self._charge_capacity = discharge_rate * duration
        self._efficiency = roundtrip_efficiency**0.5

    def get_hourly_capacity(
        self,
        start_hour: datetime,
        end_hour: datetime,
        net_hourly_capacity: xr.DataArray,
    ):
        # slice net capacity
        net_hourly_capacity = net_hourly_capacity.sel(time=slice(start_hour, end_hour))

        # initialize full storage unit
        current_charge = self._charge_capacity
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


# ENERGY SYSTEM


class EnergySystem:
    """Class responsible for managing energy units."""

    unit_dispatch_order = {
        DemandUnit: 0,
        StaticUnit: 1,
        ConstantDemandUnit: 2,
        StochasticUnit: 3,
        StorageUnit: 4,
    }

    def __init__(self):
        self._energy_units = []

    @property
    def size(self):
        return len(self._energy_units)

    @property
    def capacity(self):
        return sum([u.nameplate_capacity for u in self._energy_units])

    @property
    def energy_units(self):
        return tuple(self._energy_units)

    def add_unit(self, energy_unit: EnergyUnit):
        # check for duplicates
        if energy_unit.id in [u.id for u in self._energy_units]:
            raise RuntimeError("Duplicate unit placed in energy system.")

        # add unit to internal list
        self._energy_units.append(energy_unit)

        # sort by energy unit type
        self._energy_units.sort(key=lambda u: self.unit_dispatch_order[type(u)])

    def remove_unit(self, energy_unit: EnergyUnit):
        self._energy_units.remove(energy_unit)

    def add_system(self, other: EnergySystem):
        for energy_unit in other._energy_units:
            self.add_unit(energy_unit)

    def remove_system(self, other: EnergySystem):
        for energy_unit in other._energy_units:
            self.remove_unit(energy_unit)

    def get_hourly_capacity_by_unit(
        self,
        start_hour: datetime,
        end_hour: datetime,
        net_hourly_capacity: xr.DataArray = None,
    ):

        # initialize matrices
        time_stamps = xr.date_range(start_hour, end_hour, freq="H", inclusive="both")
        hourly_capacity_by_unit = xr.DataArray(
            data=np.zeros((self.size, len(time_stamps))),
            coords=dict(
                energy_unit=[u.id for u in self._energy_units], time=time_stamps
            ),
        )

        # check for passed net capacity series
        if net_hourly_capacity is None:
            net_hourly_capacity = xr.DataArray(
                data=np.zeros(len(time_stamps)), coords=dict(time=time_stamps)
            )

        # iterate through dispatch
        for energy_unit in self._energy_units:
            hourly_capacity_inst = energy_unit.get_hourly_capacity(
                start_hour, end_hour, net_hourly_capacity
            )
            hourly_capacity_by_unit.loc[energy_unit.id] = hourly_capacity_inst
            net_hourly_capacity += hourly_capacity_inst

        return hourly_capacity_by_unit
