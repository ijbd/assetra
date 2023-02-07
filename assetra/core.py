from __future__ import annotations
from abc import abstractmethod, ABC
from logging import getLogger

# external
from numpy.typing import ArrayLike
import numpy as np

log = getLogger(__name__)

# ENERGY UNIT(S)


class EnergyUnit(ABC):
    def __init__(self, nameplate_capacity: float, is_responsive: bool):
        self._nameplate_capacity = nameplate_capacity
        self._is_responsive = is_responsive

    # READ-ONLY VARIABLES

    @property
    def nameplate_capacity(self):
        return self._nameplate_capacity

    @property
    def is_responsive(self):
        return self._is_responsive

    # METHODS

    @abstractmethod
    def get_hourly_capacity(self):
        """Returns a single instance of the hourly capacity of the
        generating unit."""
        pass


class StaticUnit(EnergyUnit):
    """Class responsible for returning capacity profile of non-stochastic units
    (i.e. system loads)."""

    def __init__(self, nameplate_capacity: float, hourly_capacity: np.ndarray):
        EnergyUnit.__init__(self, nameplate_capacity=nameplate_capacity, is_responsive=False)
        self._hourly_capacity = hourly_capacity

    def get_hourly_capacity(self, start_hour: int, end_hour: int):
        return self._hourly_capacity[start_hour:end_hour]


class DemandUnit(StaticUnit):
    """Class responsible for returning capacity profile of fixed demand units
    (i.e. system loads)."""

    def __init__(self, hourly_demand: np.ndarray):
        StaticUnit.__init__(
            self, nameplate_capacity=0, hourly_capacity=-hourly_demand
        )


class StochasticUnit(EnergyUnit):
    """Class responsible for returning capacity profile of
    stochastically-sampled units (i.e. generators)."""

    def __init__(
        self,
        nameplate_capacity: float,
        hourly_capacity: ArrayLike,
        hourly_forced_outage_rate: ArrayLike,
    ):
        # initialize base class variables
        EnergyUnit.__init__(self, nameplate_capacity=nameplate_capacity, is_responsive=False)
        # initialize stochastic specific variables
        self._hourly_capacity = hourly_capacity
        self._hourly_forced_outage_rate = hourly_forced_outage_rate

    def get_hourly_capacity(self, start_hour: int, end_hour: int):
        hourly_outage_samples = np.random.random_sample(end_hour - start_hour)
        hourly_capacity_instance = np.where(
            hourly_outage_samples
            > self._hourly_forced_outage_rate[start_hour:end_hour],
            self._hourly_capacity[start_hour:end_hour],
            0,
        )
        return hourly_capacity_instance


class StorageUnit(EnergyUnit):
    """Class responsible for returning capacity profile of state-limited
    storage units"""
    def __init__(
        self,
        charge_rate: float,
        discharge_rate: float,
        duration: float,
        roundtrip_efficiency: float,
    ):
        EnergyUnit.__init__(self, nameplate_capacity=discharge_rate, is_responsive=True)
        self._charge_rate = charge_rate
        self._discharge_rate = discharge_rate
        self._charge_capacity = discharge_rate * duration
        self._efficiency = roundtrip_efficiency**0.5

    def get_hourly_capacity(self, net_hourly_capacity: ArrayLike):
        # initialize full storage unit
        self._current_charge = self._charge_capacity

        # simulate dispatch
        hourly_capacity = np.array(
            [
                self._dispatch_storage(net_capacity)
                for net_capacity in net_hourly_capacity
            ]
        )
        
        return hourly_capacity

    def _dispatch_storage(self, net_capacity: float):
        capacity = 0
        if net_capacity < 0:
            # unmet demand
            if self._current_charge > 0:
                capacity = self._discharge_storage(-net_capacity)
        else:
            # excess capacity
            if self._current_charge < self._charge_capacity:
                capacity = self._charge_storage(net_capacity)

        return capacity

    def _charge_storage(self, excess_capacity: float):
        capacity = -min(
            self._charge_rate,
            (self._charge_capacity - self._current_charge) / self._efficiency,
            excess_capacity,
        )
        self._current_charge -= capacity * self._efficiency

        return capacity

    def _discharge_storage(self, unmet_demand: float):
        capacity = min(
            self._discharge_rate / self._efficiency,
            self._current_charge,
            unmet_demand / self._efficiency,
        )
        self._current_charge -= capacity

        return capacity * self._efficiency


# ENERGY SYSTEM


class EnergySystem:
    """Class responsible for managing energy units."""

    def __init__(self):
        self._energy_units = []

    @property
    def size(self):
        return len(self._energy_units)

    @property
    def capacity(self):
        return sum([u.nameplate_capacity for u in self._energy_units])

    def add_unit(self, energy_unit: EnergyUnit):
        self._energy_units.append(energy_unit)

    def remove_unit(self, energy_unit: EnergyUnit):
        self._energy_units.remove(energy_unit)

    def get_hourly_capacity_by_unit(self, start_hour: int, end_hour: int):
        """Returns the hourly capacity of each generating unit
        in the energy system."""
        hourly_capacity_matrix = np.zeros((self.size, end_hour - start_hour))
        hourly_net_capacity = np.zeros(end_hour - start_hour)
        for i, energy_unit in enumerate(self._energy_units):
            if not energy_unit.is_responsive:
                hourly_capacity_matrix[i] = energy_unit.get_hourly_capacity(
                    start_hour, end_hour
                )
            else:
                hourly_capacity_matrix[i] = energy_unit.get_hourly_capacity(
                    hourly_net_capacity
                )
                
            hourly_net_capacity += hourly_capacity_matrix[i]

        return hourly_capacity_matrix
