from logging import getLogger
from abc import ABC, abstractmethod

from assetra.units import (
    RESPONSIVE_UNIT_TYPES,
    NONRESPONSIVE_UNIT_TYPES,
    StaticUnit,
    StochasticUnit,
    HydroUnit,
    StorageUnit,
)
from assetra.system import EnergySystem
from assetra.simulation import ProbabilisticSimulation
from assetra.metrics import ResourceAdequacyMetric

import xarray as xr

LOG = getLogger(__name__)

# unit types that can be accredited as resources of a base system (exact
# types: subclasses such as DemandUnit are not supported)
_SUPPORTED_RESOURCE_TYPES = (StaticUnit, StochasticUnit, HydroUnit, StorageUnit)


class ResourceContributionMetric(ABC):
    """Class responsible for quantifying resource contributions
    to an energy system

    Args:
        energy_system (EnergySystem): Base system to which resources are added
        simulation (ProbabilisticSimulation): Instantiated simulation object
            whose parameters (i.e. time range and trial size are used
            throughout the ELCC calculation)
        resource_adequacy_metric: Class derived
            from ResourceAdequacyMetric to use throughout the ELCC simulation
            (e.g. ExpectedUnservedEnergy, LossOfLoadHours)
    """

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
    ):
        self._original_energy_system = energy_system
        self._simulation = simulation
        self._resource_adequacy_metric = resource_adequacy_metric

    @abstractmethod
    def evaluate(self, addition: EnergySystem) -> float:
        """Return resource contribution of addition to the energy system

        Args:
            addition (EnergySystem): Energy system to add (possibly a single
            unit)

        Returns:
            float: Quantified resource contribution of addition to the energy
                system.
        """


def _zero_net_matrix(simulation: ProbabilisticSimulation) -> xr.DataArray:
    """Return a net hourly capacity matrix of zeros for a simulation.

    A copy of the simulation runs an empty energy system, so the matrix has
    exactly the trial and time coordinates that the simulation produces.

    Args:
        simulation (ProbabilisticSimulation): Simulation whose time range and
            trial size are used. It is not modified.

    Returns:
        xr.DataArray: Matrix of zeros with dimensions (trial, time).
    """
    empty_simulation = simulation.copy()
    empty_simulation.assign_energy_system(EnergySystem({}))
    empty_simulation.run()
    zero_matrix = empty_simulation.net_hourly_capacity_matrix

    # unit outputs are added onto this matrix by position (with .values)
    assert zero_matrix.dims == ("trial", "time"), (
        "Unexpected net hourly capacity matrix dimensions: "
        + str(zero_matrix.dims)
    )
    return zero_matrix


def _resource_unit_ids(
    energy_system: EnergySystem, resources: EnergySystem
) -> dict[type, list[int]]:
    """Return the ids of the resource units, matched to the base system.

    Unit datasets are matched by their exact unit type (the dict key, with
    no subclass matching) and units by the ids in their energy_unit
    coordinate. The resources only identify units: the base system's data
    is used for everything else.

    Args:
        energy_system (EnergySystem): Base system containing the resources.
        resources (EnergySystem): Units of the base system to accredit
            together as one resource.

    Returns:
        dict[type, list[int]]: Resource unit ids by unit type, in the order
            in which the units appear in the base system's unit dataset.

    Raises:
        TypeError: Resources are not an EnergySystem.
        ValueError: Resources are empty, hold an unsupported unit type, hold
            hydro units while a hydro unit dataset has no unit ids, or hold
            a unit that is not in the base system.
    """
    # check for a non-empty energy system
    if not isinstance(resources, EnergySystem):
        message = (
            "Resources must be an EnergySystem, got "
            + type(resources).__name__
        )
        LOG.error(message)
        raise TypeError(message)
    if resources.size == 0:
        message = (
            "Resources are empty; they must hold units of the base system"
        )
        LOG.error(message)
        raise ValueError(message)

    # check unit types (exact types) before any ids
    for unit_type in resources.unit_datasets:
        if unit_type not in _SUPPORTED_RESOURCE_TYPES:
            message = (
                f"Resources of type {unit_type.__name__} are not supported; "
                "supported types are "
                + ", ".join(t.__name__ for t in _SUPPORTED_RESOURCE_TYPES)
            )
            LOG.error(message)
            raise ValueError(message)

    # check hydro unit ids, without which positions would stand in for ids
    if HydroUnit in resources.unit_datasets:
        for name, system in (
            ("base system", energy_system),
            ("resources", resources),
        ):
            unit_dataset = system.unit_datasets.get(HydroUnit)
            if unit_dataset is None or "energy_unit" in unit_dataset.coords:
                continue
            message = (
                f"HydroUnit dataset of the {name} has no unit ids (no "
                "energy_unit coordinate); rebuild the system from its units "
                "with this version of assetra"
            )
            LOG.error(message)
            raise ValueError(message)

    # check that every resource unit is in the base system under its type
    resource_ids = dict()
    for unit_type, unit_dataset in resources.unit_datasets.items():
        base_dataset = energy_system.unit_datasets.get(unit_type)
        base_ids = (
            []
            if base_dataset is None
            else [int(i) for i in base_dataset.energy_unit.values]
        )
        ids = [int(i) for i in unit_dataset.energy_unit.values]
        base_id_set = set(base_ids)
        for unit_id in ids:
            if unit_id not in base_id_set:
                message = (
                    f"{unit_type.__name__} with id {unit_id} is not in the "
                    "base system"
                )
                LOG.error(message)
                raise ValueError(message)

        # follow the base system's order, whatever the order of resources
        selected_ids = set(ids)
        resource_ids[unit_type] = [i for i in base_ids if i in selected_ids]

    return resource_ids


def _replay_non_responsive(
    energy_system: EnergySystem,
    resource_ids: dict[type, list[int]],
    simulation: ProbabilisticSimulation,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Replay the non-responsive dispatch of a simulation run, splitting out
    the resource's own output.

    Unit datasets are dispatched onto a running net in the order of
    NONRESPONSIVE_UNIT_TYPES, read at call time, as
    ProbabilisticSimulation.run does. A unit dataset without resource units
    is dispatched whole. A static or stochastic unit dataset with resource
    units is sampled in two calls, the rest of the fleet first (if any) and
    then the resource units, so the resource's output comes from the same
    draws as the net (common random numbers). Hydro units are dispatched one
    at a time in dataset order, each against the running net, which is the
    sequential algorithm HydroUnit uses for a whole dataset.

    Args:
        energy_system (EnergySystem): Base system. It is not modified.
        resource_ids (dict[type, list[int]]): Resource unit ids by unit
            type, as returned by _resource_unit_ids.
        simulation (ProbabilisticSimulation): Simulation whose time range and
            trial size are used. It is not modified.

    Returns:
        tuple[xr.DataArray, xr.DataArray]: Net hourly capacity matrix after
            all non-responsive units (N_nr) and the resource's own output
            (X_T), both with dimensions (trial, time).
    """
    non_responsive_net = _zero_net_matrix(simulation)
    resource_output = xr.zeros_like(non_responsive_net)

    for unit_type in NONRESPONSIVE_UNIT_TYPES:
        if unit_type not in energy_system.unit_datasets:
            continue
        unit_dataset = energy_system.unit_datasets[unit_type]
        ids = resource_ids.get(unit_type)

        if not ids:
            # no resource units: dispatch the whole dataset
            non_responsive_net += unit_type.get_probabilistic_capacity_matrix(
                unit_dataset, non_responsive_net
            ).values
        elif unit_type is HydroUnit:
            # one unit at a time, capturing the resource units' dispatch
            for i, unit_id in enumerate(unit_dataset.energy_unit.values):
                unit_output = unit_type.get_probabilistic_capacity_matrix(
                    unit_dataset.isel(energy_unit=slice(i, i + 1)),
                    non_responsive_net,
                ).values
                non_responsive_net += unit_output
                if int(unit_id) in ids:
                    resource_output += unit_output
        else:
            # static or stochastic: the rest of the fleet, then the resource
            rest_dataset = unit_dataset.drop_sel(energy_unit=ids)
            if rest_dataset.sizes["energy_unit"] > 0:
                non_responsive_net += (
                    unit_type.get_probabilistic_capacity_matrix(
                        rest_dataset, non_responsive_net
                    ).values
                )
            unit_output = unit_type.get_probabilistic_capacity_matrix(
                unit_dataset.sel(energy_unit=ids), non_responsive_net
            ).values
            non_responsive_net += unit_output
            resource_output += unit_output

    return non_responsive_net, resource_output


class EffectiveLoadCarryingCapability(ResourceContributionMetric):
    """Class responsible for quantifying resource contribution to an energy
    system using the effective load-carrying capability metric:

    "[The] amount by which the system's load can increase when the resource is
    added to the system while maintaining the same system adequacy."

    https://gridops.epri.com/Adequacy/metrics#Effective_Load_Carrying_Capability_.28ELCC.29

    Args:
        energy_system (EnergySystem) : Base system to which resources are added
        simulation (ProbabilisticSimulation) : Instantiated simulation object
            whose parameters (i.e. time range and trial size are used
            throughout the ELCC calculation)
        resource_adequacy_metric (type[ResourceAdequacyMetric') : Class derived
            from ResourceAdequacyMetric to use throughout the ELCC simulation
            (e.g. ExpectedUnservedEnergy, LossOfLoadHours)
    """

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
    ):
        """The ELCC computes resource adequacy many times iteratively. In many
        instances, it is neither necessary nor desirable to re-run the
        probabilistic simulation for a set of units more than once. For
        so-called non-responsive units (static and stochastic units) which do
        not respond to system conditions, a single probabilistic simulation
        is sufficient. For responsive units, including storage, changing system
        conditions necessitate re-evaluation of probabilistic capacity
        contributions.

        To account for this classification in the iterative resource adequacy
        calculations, the ELCC decomposes energy systems into their responsive
        and non-responsive components. In each iteration, combinations of the
        component energy systems' probabilistic simulations are "chained"
        together depending on which need to be recomputed.

        For example, to establish the baseline resource adequacy (as in the
        init function), both the original responsive system and the
        original non-responsive systems are evaluated. Programmatically, the
        non-responsive system is evaluated first. Then the resulting
        net hourly capacity matrix is fed into the responsive system.

        When a new system is added, it is again decomposed into responsive and
        non-responsive subsystems. The original non-responsive system does not
        need to be re-evaluated, so its already-simulated probabilistic net
        hourly capacity is fed into the new non-responsive system which is fed
        into the new and original responsive systems respectively.

        In this way, non-responsive units are only evaluated once while
        responsive units are able to respond to changes in system conditions.
        """
        ResourceContributionMetric.__init__(
            self, energy_system, simulation, resource_adequacy_metric
        )

        # decompose system into responsive and non-responsive components
        # non-responsive simulation
        self._original_system_non_responsive = (
            self._original_energy_system.get_system_by_type(
                NONRESPONSIVE_UNIT_TYPES
            )
        )
        self._original_non_responsive_simulation = self._simulation.copy()
        self._original_non_responsive_simulation.assign_energy_system(
            self._original_system_non_responsive
        )

        # responsive simulation
        self._original_system_responsive = (
            self._original_energy_system.get_system_by_type(
                RESPONSIVE_UNIT_TYPES
            )
        )
        self._original_responsive_simulation = self._simulation.copy()
        self._original_responsive_simulation.assign_energy_system(
            self._original_system_responsive
        )

        # run chained simulation
        self._original_non_responsive_simulation.run()
        self._original_responsive_simulation.run(
            self._original_non_responsive_simulation.net_hourly_capacity_matrix
        )
        self._resource_adequacy_model = self._resource_adequacy_metric(
            self._original_responsive_simulation
        )
        self._original_resource_adequacy = (
            self._resource_adequacy_model.evaluate()
        )
        LOG.info(
            "Original resource adequacy: "
            + str(self._original_resource_adequacy)
        )

        if self._original_resource_adequacy == 0:
            LOG.error("Invalid ELCC calculation for system with no risk")
            raise RuntimeWarning()

        # save intermediate steps
        self._original_net_capacity_matrix = (
            self._original_responsive_simulation.net_hourly_capacity_matrix
        )
        self._intermediate_net_capacity_matrices = []

    @property
    def original_net_capacity_matrix(self) -> xr.DataArray:
        """Return the net hourly capacity matrix of the base system"""
        # TODO test
        return self._original_net_capacity_matrix.copy()

    @property
    def intermediate_net_capacity_matrices(
        self,
    ) -> tuple[tuple[float, xr.DataArray]]:
        """Return intermediate net capacity matrices for each step of the ELCC
        calculation. Each element is a tuple composed of the amount of added
        constant load and the net hourly capacity matrix corresponding to that
        step.
        """
        # TODO test
        return tuple(m for m in self._intermediate_net_capacity_matrices)

    def evaluate(
        self,
        addition: EnergySystem,
        additional_demand_resolution_pct: float = 0.01,
    ) -> float:
        """Return the ELCC of an addition to the energy system.

        Args:
            addition (EnergySystem): Energy system to add (possibly a
                single unit).
            additional_demand_resolution_pct (float, optional): Resolution of
                added demand to find as percent of added nameplate capacity.
                Defaults to 0.01 (1%). E.g. for a 100 MW addition, the ELCC
                will be found within 1 MW.

        Returns:
            float: Amount of added constant load in units of power.
        """

        # reset intermediate net capacities
        self._intermediate_net_capacity_matrices = []

        # decompose system into responsive and non-responsive components
        # non-responsive simulation
        additional_system_non_responsive = addition.get_system_by_type(
            NONRESPONSIVE_UNIT_TYPES
        )
        additional_non_responsive_simulation = self._simulation.copy()
        additional_non_responsive_simulation.assign_energy_system(
            additional_system_non_responsive
        )

        # responsive simulation
        additional_system_responsive = addition.get_system_by_type(
            RESPONSIVE_UNIT_TYPES
        )
        additional_responsive_simulation = self._simulation.copy()
        additional_responsive_simulation.assign_energy_system(
            additional_system_responsive
        )

        # run non-responsive_simulation
        additional_non_responsive_simulation.run()

        # get non-responsive net hourly capacity
        non_responsive_net_hourly_capacity_matrix = (
            self._original_non_responsive_simulation.net_hourly_capacity_matrix
            + additional_non_responsive_simulation.net_hourly_capacity_matrix
        )

        # add load
        additional_demand_upper_bound = addition.system_capacity
        additional_demand_lower_bound = 0
        additional_demand_resolution = (
            additional_demand_upper_bound - additional_demand_lower_bound
        )
        additional_demand = (
            additional_demand_lower_bound + additional_demand_resolution / 2
        )

        # run chained responsive simulation
        additional_responsive_simulation.run(
            non_responsive_net_hourly_capacity_matrix - additional_demand
        )
        self._original_responsive_simulation.run(
            additional_responsive_simulation.net_hourly_capacity_matrix
        )
        self._intermediate_net_capacity_matrices.append(
            (
                additional_demand,
                self._original_responsive_simulation.net_hourly_capacity_matrix,
            )
        )

        # update resource adequacy
        new_resource_adequacy = self._resource_adequacy_model.evaluate()

        # printout
        LOG.info("Additional demand: " + str(round(additional_demand)))
        LOG.info("Resource adequacy: " + str(new_resource_adequacy))

        # iterate until convergence
        iteration = 0

        while (
            additional_demand_resolution / addition.system_capacity
        ) > additional_demand_resolution_pct:
            # iterate until original resource adequacy level is met
            if new_resource_adequacy > self._original_resource_adequacy:
                # if over-reliable, add load
                additional_demand_upper_bound = additional_demand
            else:
                # if under-reliable, remove load
                additional_demand_lower_bound = additional_demand
            # add demand
            additional_demand_resolution = (
                additional_demand_upper_bound - additional_demand_lower_bound
            )
            additional_demand = (
                additional_demand_lower_bound + additional_demand_resolution / 2
            )

            # run chained responsive simulation
            additional_responsive_simulation.run(
                non_responsive_net_hourly_capacity_matrix - additional_demand
            )
            self._original_responsive_simulation.run(
                additional_responsive_simulation.net_hourly_capacity_matrix
            )
            self._intermediate_net_capacity_matrices.append(
                (
                    additional_demand,
                    self._original_responsive_simulation.net_hourly_capacity_matrix,
                )
            )

            # update resource adequacy
            new_resource_adequacy = self._resource_adequacy_model.evaluate()

            LOG.info("Additional demand: " + str(additional_demand))
            LOG.info("Resource adequacy: " + str(new_resource_adequacy))

            # update iteration count
            iteration += 1

        return float(additional_demand)
