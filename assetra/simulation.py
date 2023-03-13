from __future__ import annotations
from datetime import datetime
from logging import getLogger

# external
import numpy as np
import xarray as xr

# package
from assetra.units import NONRESPONSIVE_UNIT_TYPES, RESPONSIVE_UNIT_TYPES
from assetra.system import EnergySystem

LOG = getLogger(__name__)


class ProbabilisticSimulation:
    """Class responsible for creating/storing the Monte Carlo
    trials for EnergySystem objects.

    Args:
        start_hour (datetime) : Starting simulation hour,
            for example "2016-01-01 00:00:00"
        end_hour (datetime) : Ending simulation hour (inclusive).
        trial_size (int) : Number of simulation Monte Carlo trials.
    """

    def __init__(
        self, start_hour: datetime, end_hour: datetime, trial_size: int
    ):
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._trial_size = trial_size

        # state variables
        self._energy_system = None
        self._net_hourly_capacity_matrix = None
        self._hourly_capacity_matrix = None

    def copy(self) -> ProbabilisticSimulation:
        """Return a probabilistic simulation object with the same underlying
        parameters but no assigned system

        Returns:
            ProbabilisticSimulation: Simulation with same start hour, end hour,
                and trial size as this object."""
        return ProbabilisticSimulation(
            self._start_hour, self._end_hour, self._trial_size
        )

    def assign_energy_system(self, energy_system: EnergySystem) -> None:
        """Assign an energy system to this probabilistic simulation object.
        Nullifies the stored capacity matrices

        Args:
            energy_system (EnergySystem): Energy system to simulate.

        Raises:
            RuntimeWarning: Invalid type assigned to simulation energy system.
        """
        if not isinstance(energy_system, EnergySystem):
            LOG.warning("Invalid type assigned to simulation energy system.")
            raise RuntimeWarning()
        self._energy_system = energy_system
        self._net_hourly_capacity_matrix = None
        self._hourly_capacity_matrix = None

    @property
    def net_hourly_capacity_matrix(self) -> xr.DataArray:
        """Return the resultant net hourly capacity matrix for this simulation.
        If it does not exist, trigger a simulation run

        Returns:
            xr.DataArray: Net hourly capacity matrix with dimensions (trials,
                time) and shape (# of trials, # of hours)
        """
        if self._net_hourly_capacity_matrix is None:
            self.run()
        return self._net_hourly_capacity_matrix.copy()

    def get_hourly_capacity_matrix_by_type(
        self, unit_type: type
    ) -> xr.DataArray:
        """Return the resultant hourly capacity matrix for a unit_type.

        Hourly capacity matrices for each unit dataset are evaluated by the
        energy unit classes. The probabilistic simulation stores a copy of the
        combined hourly capacity indexed by unit type. It is not possible to
        get the hourly capacity matrix for each unit of a specific type, only
        the aggregate.

        If the hourly capacity matrix does not exist, trigger a simulation run

        Args:
            unit_type (_type_): A valid energy unit type.

        Returns:
            xr.DataArray: Hourly capacity matrix with dimensions (trials, time)
              and shape (# of trials, # of hours)
        """
        if self._hourly_capacity_matrix is None:
            self.run()
        return self._hourly_capacity_matrix.sel(unit_type=unit_type)

    def run(self, net_hourly_capacity_matrix: xr.DataArray = None) -> None:
        """Run the probabilistic simulation. This function evaluates the
        net hourly capacity matrix and hourly capacity matrix for each unit
        type. Optionally, provide a pre-existing net hourly capacity matrix
        to dispatch onto.

        An energy system should already be assigned to the simulation
        object.

        Args:
            net_hourly_capacity_matrix (xr.DataArray, optional): A net hourly
            capacity matrix from a similar simulation with the same start/end
            hours and trial size. If passed, the matrix is modified.
            Defaults to None.

        Raises:
            RuntimeWarning: Energy system not assigned to simulation object.
        """
        # check for energy system
        if not isinstance(self._energy_system, EnergySystem):
            LOG.warning("Energy system not assigned to simulation object.")
            raise RuntimeWarning()

        # setup net hourly capacity matrix
        time_stamps = xr.date_range(
            self._start_hour, self._end_hour, freq="H", inclusive="both"
        )

        # initialize net capacity matrix
        if net_hourly_capacity_matrix is not None:
            # check dimensions
            assert net_hourly_capacity_matrix.sizes["trial"] == self._trial_size
            assert net_hourly_capacity_matrix.sizes["time"] == time_stamps.size
            assert net_hourly_capacity_matrix.time[0] == time_stamps[0]
            # load net hourly capacity
            self._net_hourly_capacity_matrix = net_hourly_capacity_matrix
        else:
            self._net_hourly_capacity_matrix = xr.DataArray(
                data=np.zeros((self._trial_size, len(time_stamps))),
                coords=dict(
                    trial=np.arange(self._trial_size), time=time_stamps
                ),
            )

        # initialize capacity by unit type
        unit_types = NONRESPONSIVE_UNIT_TYPES + RESPONSIVE_UNIT_TYPES
        self._hourly_capacity_matrix = xr.DataArray(
            data=np.zeros(
                (len(unit_types), self._trial_size, len(time_stamps))
            ),
            coords=dict(
                unit_type=unit_types,
                trial=np.arange(self._trial_size),
                time=time_stamps,
            ),
        )

        # iterate through unit datasets
        for (
            unit_type,
            unit_dataset,
        ) in self._energy_system.unit_datasets.items():
            self._hourly_capacity_matrix.loc[
                unit_type
            ] = unit_type.get_probabilistic_capacity_matrix(
                unit_dataset,
                self._net_hourly_capacity_matrix,
            ).values
            self._net_hourly_capacity_matrix += (
                self._hourly_capacity_matrix.sel(unit_type=unit_type).values
            )
