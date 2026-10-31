from logging import getLogger
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import math
import numbers

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

# default perturbation step, as a fraction of the resources' nameplate capacity
_DEFAULT_STEP_FRACTION = 0.01


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


def _scale_storage_dataset(
    unit_dataset: xr.Dataset, unit_ids: list[int], scale: float
) -> xr.Dataset:
    """Return a copy of a storage unit dataset with some of its units scaled.

    The charge rate, discharge rate, charge capacity and nameplate capacity
    of the selected units are multiplied by scale. Their initial state of
    charge (a fraction, absent in datasets saved before it existed) and
    roundtrip efficiency are unchanged, so their initial energy scales too.

    Args:
        unit_dataset (xr.Dataset): Storage unit dataset. It is not modified.
        unit_ids (list[int]): Ids of the units to scale.
        scale (float): Factor applied to the selected units.

    Returns:
        xr.Dataset: New storage unit dataset, with the units in the same
            order.
    """
    selected = unit_dataset.energy_unit.isin(unit_ids)
    return unit_dataset.assign(
        {
            name: xr.where(
                selected, unit_dataset[name] * scale, unit_dataset[name]
            )
            for name in (
                "charge_rate",
                "discharge_rate",
                "charge_capacity",
                "nameplate_capacity",
            )
        }
    )


@dataclass(frozen=True)
class _MarginalBaseline:
    """Baseline of a marginal accreditation, shared by every resource
    adequacy evaluation of one evaluate() call.

    Attributes:
        step (float): Perturbation step in units of power.
        nameplate (float): Total nameplate capacity of the resources (x_T).
        non_responsive_net (xr.DataArray): Net hourly capacity matrix after
            all non-responsive units (N_nr).
        resource_output (xr.DataArray): The resources' own non-responsive
            output, from the same samples as the net (X_T).
        storage_ids (tuple[int, ...]): Ids of the resources' storage units.
        risk (float): Baseline resource adequacy (M0).
    """

    step: float
    nameplate: float
    non_responsive_net: xr.DataArray
    resource_output: xr.DataArray
    storage_ids: tuple[int, ...]
    risk: float


class _MarginalContributionMetric(ResourceContributionMetric):
    """Base class for the marginal accreditation of resources that are
    already in an energy system.

    Each evaluate() call samples the base system once and splits out the
    resources' own output, so every perturbed evaluation reuses the
    resources' own sampled outages (common random numbers). Perturbing the
    resources by a signed step scales their non-responsive output, and the
    ratings of their storage units, by 1 + step / nameplate capacity. The
    responsive units are re-dispatched in every evaluation, while the other
    non-responsive units keep their sampled output.

    Args:
        energy_system (EnergySystem): Base system containing the resources.
        simulation (ProbabilisticSimulation): Instantiated simulation object
            whose parameters (i.e. time range and trial size) are used
            throughout the calculation. It is not modified.
        resource_adequacy_metric (type[ResourceAdequacyMetric]): Class
            derived from ResourceAdequacyMetric to use throughout the
            calculation (e.g. ExpectedUnservedEnergy, LossOfLoadHours).
        perturbation_size (float | None): Step in units of power, or None
            for 1% of the resources' nameplate capacity in each evaluate()
            call.
        method (str): Computation method, one of the subclass's methods.

    Raises:
        TypeError: energy_system is not an EnergySystem, or
            resource_adequacy_metric is not a ResourceAdequacyMetric class.
        ValueError: method is not one of the subclass's methods, or
            perturbation_size is neither None nor a finite number greater
            than 0.
    """

    # computation methods accepted by the subclass
    _methods: tuple[str, ...] = ()

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
        perturbation_size: float | None,
        method: str,
    ) -> None:
        # check arguments in order; no simulation runs here
        if not isinstance(energy_system, EnergySystem):
            message = (
                "energy_system must be an EnergySystem, got "
                + type(energy_system).__name__
            )
            LOG.error(message)
            raise TypeError(message)
        if not (
            isinstance(resource_adequacy_metric, type)
            and issubclass(resource_adequacy_metric, ResourceAdequacyMetric)
        ):
            message = (
                "resource_adequacy_metric must be a class derived from "
                "ResourceAdequacyMetric (e.g. ExpectedUnservedEnergy), got "
                + repr(resource_adequacy_metric)
            )
            LOG.error(message)
            raise TypeError(message)
        if method not in self._methods:
            message = (
                "method must be one of "
                + ", ".join(repr(m) for m in self._methods)
                + f", got {method!r}"
            )
            LOG.error(message)
            raise ValueError(message)
        if perturbation_size is not None and not (
            isinstance(perturbation_size, numbers.Real)
            and math.isfinite(perturbation_size)
            and perturbation_size > 0
        ):
            message = (
                "perturbation_size must be None or a finite number greater "
                f"than 0, got {perturbation_size!r}"
            )
            LOG.error(message)
            raise ValueError(message)

        ResourceContributionMetric.__init__(
            self, energy_system, simulation, resource_adequacy_metric
        )
        self._perturbation_size = (
            None if perturbation_size is None else float(perturbation_size)
        )
        self._method = method
        self._perturbation_count = 0

    @property
    def perturbation_count(self) -> int:
        """Return the number of resource adequacy evaluations with the
        resources perturbed (the paper's N_p) in the latest evaluate() call,
        or 0 before the first call.

        The baseline and perfect-resource evaluations are not counted: in
        total, a call evaluates resource adequacy N_p times, plus once for
        the baseline, plus once (forward) or twice (central) with a perfect
        resource for MarginalReliabilityImpact.
        """
        return self._perturbation_count

    def _resolve(
        self, resources: EnergySystem
    ) -> tuple[dict[type, list[int]], float, float]:
        """Match resources to the base system and return their total
        nameplate capacity and the perturbation step.

        Args:
            resources (EnergySystem): Units of the base system to accredit
                together as one resource.

        Returns:
            tuple[dict[type, list[int]], float, float]: Resource unit ids by
                unit type (as returned by _resource_unit_ids), the total
                nameplate capacity of those units in the base system (x_T),
                and the step: perturbation_size, or 1% of x_T if None.

        Raises:
            TypeError: Resources are not an EnergySystem.
            ValueError: Resources are empty, hold an unsupported unit type,
                hold hydro units without unit ids, or hold a unit that is not
                in the base system; or their total nameplate capacity is not
                greater than 0.
        """
        resource_ids = _resource_unit_ids(
            self._original_energy_system, resources
        )

        # the base system's data is used: resources only identify the units
        nameplate = sum(
            float(
                self._original_energy_system.unit_datasets[unit_type][
                    "nameplate_capacity"
                ]
                .sel(energy_unit=ids)
                .sum()
            )
            for unit_type, ids in resource_ids.items()
            if ids
        )
        if not nameplate > 0:
            message = (
                "Resources must have a total nameplate capacity greater than "
                f"0, got {nameplate}"
            )
            LOG.error(message)
            raise ValueError(message)

        if self._perturbation_size is None:
            step = _DEFAULT_STEP_FRACTION * nameplate
        else:
            step = self._perturbation_size
        return resource_ids, nameplate, step

    def _baseline(
        self,
        resource_ids: dict[type, list[int]],
        nameplate: float,
        step: float,
    ) -> _MarginalBaseline:
        """Sample the base system once and evaluate its resource adequacy.

        Args:
            resource_ids (dict[type, list[int]]): Resource unit ids by unit
                type, as returned by _resolve.
            nameplate (float): Total nameplate capacity of the resources.
            step (float): Perturbation step in units of power.

        Returns:
            _MarginalBaseline: Baseline shared by every resource adequacy
                evaluation of the evaluate() call.

        Raises:
            ValueError: The resources' output is negative in some trial and
                hour.
            RuntimeWarning: The base system has no risk.
        """
        non_responsive_net, resource_output = _replay_non_responsive(
            self._original_energy_system, resource_ids, self._simulation
        )
        if (resource_output.values < 0).any():
            message = (
                "Resource output is negative in some hours; only supply "
                "resources can be accredited (model demand as a DemandUnit, "
                "not as a StaticUnit with negative capacity)"
            )
            LOG.error(message)
            raise ValueError(message)

        partial = _MarginalBaseline(
            step=step,
            nameplate=nameplate,
            non_responsive_net=non_responsive_net,
            resource_output=resource_output,
            storage_ids=tuple(resource_ids.get(StorageUnit, ())),
            risk=math.nan,  # not evaluated yet
        )
        # without any perturbation, M(0, 0) equals P(0)
        risk = self._perfect_risk(partial, 0.0)
        LOG.info("Original resource adequacy: " + str(risk))
        if risk == 0:
            message = (
                "Invalid marginal accreditation for a base system with no risk"
            )
            LOG.error(message)
            raise RuntimeWarning(message)

        return replace(partial, risk=risk)

    def _risk(
        self, baseline: _MarginalBaseline, delta: float, load: float = 0.0
    ) -> float:
        """Return the resource adequacy with the resources perturbed by a
        signed step and a constant load added, M(delta, load).

        The resources' non-responsive output and storage units are scaled by
        1 + delta / nameplate capacity, and the load is subtracted after the
        non-responsive units, before the responsive units are re-dispatched.

        Args:
            baseline (_MarginalBaseline): Baseline of the evaluate() call.
            delta (float): Signed perturbation step in units of power.
            load (float, optional): Constant load added in every trial and
                hour, in units of power. Defaults to 0.

        Returns:
            float: Evaluated resource adequacy metric.
        """
        fraction = delta / baseline.nameplate
        return self._responsive_risk(
            baseline.non_responsive_net
            + fraction * baseline.resource_output
            - load,
            baseline.storage_ids,
            1 + fraction,
        )

    def _perfect_risk(
        self, baseline: _MarginalBaseline, delta: float
    ) -> float:
        """Return the resource adequacy with a perfect resource of a signed
        size added in every trial and hour, P(delta).

        Args:
            baseline (_MarginalBaseline): Baseline of the evaluate() call.
            delta (float): Signed size of the perfect resource in units of
                power.

        Returns:
            float: Evaluated resource adequacy metric.
        """
        return self._responsive_risk(baseline.non_responsive_net + delta)

    def _responsive_risk(
        self,
        net_hourly_capacity_matrix: xr.DataArray,
        storage_ids: tuple[int, ...] = (),
        scale: float = 1.0,
    ) -> float:
        """Dispatch the base system's responsive units onto a net hourly
        capacity matrix and return the resulting resource adequacy.

        Args:
            net_hourly_capacity_matrix (xr.DataArray): Net hourly capacity
                matrix after the non-responsive units. It is not modified.
            storage_ids (tuple[int, ...], optional): Ids of the storage units
                to scale. Defaults to none.
            scale (float, optional): Factor applied to those storage units.
                Defaults to 1.

        Returns:
            float: Evaluated resource adequacy metric.
        """
        unit_datasets = dict()
        for (
            unit_type,
            unit_dataset,
        ) in self._original_energy_system.unit_datasets.items():
            if unit_type not in RESPONSIVE_UNIT_TYPES:
                continue
            if unit_type is StorageUnit and storage_ids:
                unit_dataset = _scale_storage_dataset(
                    unit_dataset, list(storage_ids), scale
                )
            unit_datasets[unit_type] = unit_dataset

        simulation = self._simulation.copy()
        simulation.assign_energy_system(EnergySystem(unit_datasets))
        # run() dispatches onto the matrix it is given, so give it a copy
        simulation.run(net_hourly_capacity_matrix.copy())
        return float(self._resource_adequacy_metric(simulation).evaluate())


class MarginalReliabilityImpact(_MarginalContributionMetric):
    """Class responsible for accrediting resources that are already in an
    energy system with their marginal reliability impact (MRI): the change
    in resource adequacy when the resources are enlarged by a small step,
    relative to the change when a perfect resource of the same size is
    added.

    Q. Zhang, F. Zhao, G. Stephen, C. Singh, L. Xie, "A Gradient-Based
    Capacity Accreditation Framework in Resource Adequacy: Formulation,
    Computation, and Practical Implications," arXiv:2601.22087, 2026.

    The resources keep their own sampled outages: every perturbed
    simulation of an evaluate() call reuses that call's samples (common
    random numbers). Enlarging the resources by a step scales their
    non-responsive output, and the charge rate, discharge rate, charge
    capacity and nameplate capacity of their storage units, by
    1 + step / nameplate capacity. The perfect resource adds the step in
    every trial and hour. Responsive units are re-dispatched in every
    simulation. Non-responsive units are not: units outside the resources
    keep their sampled output, and hydro units in the resources are not
    re-dispatched against each other's scaled output.

    The method keyword selects the finite difference, where M is the
    resource adequacy with the resources enlarged by a signed step, P with
    a perfect resource of that size added, M0 the baseline, and N_p the
    number of simulations with the resources perturbed:

    - "forward" (default): (M(+step) - M0) / (P(+step) - M0), N_p = 1.
    - "central": (M(+step) - M(-step)) / (P(+step) - P(-step)), N_p = 2.
      The step must not exceed the resources' nameplate capacity.

    Unlike EffectiveLoadCarryingCapability, which returns power, evaluate()
    returns a dimensionless accreditation factor. The accredited capacity
    is the factor times the resources' nameplate capacity
    (resources.system_capacity). The factor is returned as computed, so it
    can be negative or above 1 for storage.

    Each evaluate() call samples the system afresh (seed numpy's random
    number generator for reproducible results), and nothing is cached
    between calls: every call runs N_p perturbed simulations of the
    responsive units, plus 1 for the baseline M0, plus 1 (forward) or 2
    (central) for the perfect resource.

    The default step is 1% of the resources' nameplate capacity. For large
    fleets, an absolute step such as the paper's 10 MW
    (perturbation_size=10) can be used instead. Counting metrics such as
    LossOfLoadHours and LossOfLoadDays often do not respond to a 1% step;
    use a larger step or ExpectedUnservedEnergy.

    Args:
        energy_system (EnergySystem): Base system containing the resources
            to accredit.
        simulation (ProbabilisticSimulation): Instantiated simulation object
            whose parameters (i.e. time range and trial size) are used
            throughout the calculation. It is not modified.
        resource_adequacy_metric (type[ResourceAdequacyMetric]): Class
            derived from ResourceAdequacyMetric to use throughout the
            calculation (e.g. ExpectedUnservedEnergy, LossOfLoadHours). Any
            metric class is accepted.
        perturbation_size (float | None, optional): Step in units of power.
            Defaults to None, for 1% of the nameplate capacity of the
            resources passed to evaluate().
        method (str, optional): "forward" or "central". Defaults to
            "forward".

    Attributes:
        perturbation_count (int): Number of simulations with the resources
            perturbed (N_p) in the latest evaluate() call, or 0 before the
            first call. The baseline and perfect-resource simulations are
            not counted: in total, a call evaluates resource adequacy
            N_p + 2 (forward) or N_p + 3 (central) times.

    Raises:
        TypeError: energy_system is not an EnergySystem, or
            resource_adequacy_metric is not a ResourceAdequacyMetric class.
        ValueError: method is neither "forward" nor "central", or
            perturbation_size is neither None nor a finite number greater
            than 0.
    """

    _methods = ("forward", "central")

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
        perturbation_size: float | None = None,
        method: str = "forward",
    ):
        _MarginalContributionMetric.__init__(
            self,
            energy_system,
            simulation,
            resource_adequacy_metric,
            perturbation_size,
            method,
        )

    def evaluate(self, resources: EnergySystem) -> float:
        """Return the marginal reliability impact of resources that are
        already in the energy system.

        Args:
            resources (EnergySystem): Units of the base system to accredit
                together as one resource, e.g. a system built from some of
                its units, or energy_system.get_system_by_type(StorageUnit).
                Units are matched by exact unit type and id, and the base
                system's data is used.

        Returns:
            float: Dimensionless accreditation factor. Multiplied by the
                resources' nameplate capacity, it gives their accredited
                capacity in units of power.

        Raises:
            TypeError: Resources are not an EnergySystem.
            ValueError: Resources are empty, hold a unit type other than
                StaticUnit, StochasticUnit, HydroUnit and StorageUnit, hold
                hydro units without unit ids, or hold a unit that is not in
                the base system; their total nameplate capacity is not
                greater than 0; perturbation_size exceeds that nameplate
                capacity for central differences; or their output is
                negative in some trial and hour.
            RuntimeWarning: The base system has no risk, or the metric does
                not respond to a perfect resource of the step's size.
        """
        self._perturbation_count = 0
        resource_ids, nameplate, step = self._resolve(resources)
        central = self._method == "central"
        if central and step > nameplate:
            message = (
                f"perturbation_size ({step}) must not exceed the resources' "
                f"nameplate capacity ({nameplate}) for central differences"
            )
            LOG.error(message)
            raise ValueError(message)
        baseline = self._baseline(resource_ids, nameplate, step)

        # perfect resource (denominator); forward differences use M0 below
        perfect_upper = self._perfect_risk(baseline, step)
        if central:
            perfect_lower = self._perfect_risk(baseline, -step)
        else:
            perfect_lower = baseline.risk
        denominator = perfect_upper - perfect_lower
        if denominator == 0:
            message = (
                f"{self._resource_adequacy_metric.__name__} does not respond "
                f"to a perfect resource of size {step}; increase "
                "perturbation_size or use ExpectedUnservedEnergy"
            )
            LOG.error(message)
            raise RuntimeWarning(message)

        # resources perturbed by the step (numerator), counted as N_p
        self._perturbation_count += 1
        upper = self._risk(baseline, step)
        if central:
            self._perturbation_count += 1
            lower = self._risk(baseline, -step)
        else:
            lower = baseline.risk

        factor = (upper - lower) / denominator
        LOG.info("Marginal reliability impact: " + str(factor))
        return factor


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
