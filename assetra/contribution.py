from logging import getLogger
from abc import ABC, abstractmethod
from collections.abc import Callable
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
from assetra.metrics import (
    ResourceAdequacyMetric,
    ExpectedUnservedEnergy,
    LossOfLoadHours,
    LossOfLoadDays,
)

import xarray as xr

LOG = getLogger(__name__)

# unit types that can be accredited as resources of a base system (exact
# types: subclasses such as DemandUnit are not supported)
_SUPPORTED_RESOURCE_TYPES = (StaticUnit, StochasticUnit, HydroUnit, StorageUnit)

# default perturbation step, as a fraction of the resources' nameplate capacity
_DEFAULT_STEP_FRACTION = 0.01

# caps of the root search: doublings of its bracket in each direction, and
# secant iterations
_MAX_DOUBLINGS = 10
_MAX_ITERATIONS = 100

# metrics accepted by the marginal ELCC, whose risk never decreases as load
# is added (exact types: a subclass may not stay monotone)
_ELCC_METRICS = (ExpectedUnservedEnergy, LossOfLoadHours, LossOfLoadDays)


class ResourceContributionMetric(ABC):
    """Class responsible for quantifying resource contributions
    to an energy system, either of an addition to the system
    (EffectiveLoadCarryingCapability) or of resources that are already in it
    (MarginalEffectiveLoadCarryingCapability, MarginalReliabilityImpact)

    Args:
        energy_system (EnergySystem): Base system to which resources are
            added, or that already contains the resources to accredit
        simulation (ProbabilisticSimulation): Instantiated simulation object
            whose parameters (i.e. time range and trial size are used
            throughout the calculation)
        resource_adequacy_metric: Class derived
            from ResourceAdequacyMetric to use throughout the simulation
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
        """Return resource contribution to the energy system of an addition,
        or of resources that are already in it

        Args:
            addition (EnergySystem): Energy system to quantify the
                contribution of: an addition (EffectiveLoadCarryingCapability)
                or resources already in the base system (marginal
                accreditation metrics), possibly a single unit. The marginal
                accreditation metrics name this argument resources.

        Returns:
            float: Quantified resource contribution to the energy system: in
                units of power for EffectiveLoadCarryingCapability, a
                dimensionless accreditation factor for the marginal
                accreditation metrics.
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
    unit_dataset: xr.Dataset,
    unit_ids: list[int],
    delta: float,
    nameplate: float,
) -> xr.Dataset:
    """Return a copy of a storage unit dataset with some of its units
    enlarged by a signed step.

    The charge rate, discharge rate, charge capacity and nameplate capacity
    of the selected units are multiplied by k = 1 + delta / nameplate,
    computed multiply-first as value * (nameplate + delta) / nameplate: a
    rating that should reach a round value does so exactly on integer data,
    and k = 0 gives exactly 0. Their initial state of charge (a fraction,
    absent in datasets saved before it existed) and roundtrip efficiency
    are unchanged, so their initial energy scales too.

    Args:
        unit_dataset (xr.Dataset): Storage unit dataset. It is not modified.
        unit_ids (list[int]): Ids of the units to enlarge.
        delta (float): Signed step in units of power.
        nameplate (float): Total nameplate capacity to which the step is
            relative.

    Returns:
        xr.Dataset: New storage unit dataset, with the units in the same
            order.
    """
    selected = unit_dataset.energy_unit.isin(unit_ids)
    return unit_dataset.assign(
        {
            name: xr.where(
                selected,
                unit_dataset[name] * (nameplate + delta) / nameplate,
                unit_dataset[name],
            )
            for name in (
                "charge_rate",
                "discharge_rate",
                "charge_capacity",
                "nameplate_capacity",
            )
        }
    )


def _find_largest_root(
    g: Callable[[float], float],
    upper: float,
    tolerance: float,
    method: str,
    guaranteed: bool,
    zero_tolerance: float,
    exact_root: bool,
) -> tuple[float, int]:
    """Find the largest root of a non-decreasing function, by bisection or a
    safeguarded secant.

    The root is the largest c with g(c) <= zero_tolerance, searched for from
    the bracket [0, upper]. A value of g within zero_tolerance of 0 counts as
    0, and such ties move the lower end of the bracket.

    Unless the bracket is guaranteed and the method is bisection (which then
    evaluates neither end), g is evaluated at 0 and then at upper, and the
    bracket is set up as follows:

    - If g is within zero_tolerance of 0 at both ends, g does not respond and
      RuntimeWarning is raised.
    - With exact_root, a root at 0 is returned.
    - While g at the lower end is above zero_tolerance, the bracket expands
      downward: its lower end becomes its upper end, and the new lower end is
      below it by upper, 2 * upper, 4 * upper, and so on.
    - While g at the upper end is below -zero_tolerance, the bracket expands
      upward: its upper end becomes its lower end, and the new upper end is
      twice as far from 0.
    - Each expansion is capped at _MAX_DOUBLINGS doublings, past which
      RuntimeWarning is raised.
    - If g at the upper end is then within zero_tolerance of 0, that end is
      returned: either the exact root, or a plateau where g is 0, which the
      search does not cross.

    Bisection then halves the bracket and returns the midpoint of the final
    bracket. The secant takes each next point from the secant line through
    the two most recent points. As in Dekker's method, it takes a bisection
    step instead when the denominator is 0, when the point is not strictly
    inside the bracket, or when the previous step did not halve the bracket,
    which bounds the cost at about twice that of bisection. A step from the
    latest point shorter than tolerance / 2 becomes a step of tolerance / 2
    towards the far end of the bracket, so that the bracket closes from both
    sides. Both return the midpoint of the bracket once its width is at most
    tolerance, or once it cannot be split in floating point (its midpoint is
    not strictly inside it), which is as tight as it can get when tolerance is
    finer than the spacing of floating-point numbers at the root. The secant
    also stops, with exact_root, as soon as g is within zero_tolerance of 0,
    returning that point. It is capped at _MAX_ITERATIONS iterations, past
    which RuntimeWarning is raised.

    Args:
        g (Callable[[float], float]): Non-decreasing function. Every call
            counts as one evaluation.
        upper (float): Upper end of the initial bracket [0, upper], greater
            than 0, which is also the first step of its expansion.
        tolerance (float): Width of the bracket at which the search stops, if
            floating point allows a bracket that narrow.
        method (str): "bisection" or "secant".
        guaranteed (bool): Whether g(0) <= zero_tolerance <= g(upper) is
            guaranteed, so that bisection can start from [0, upper] without
            evaluating g at either end. The secant evaluates both ends
            whatever the value.
        zero_tolerance (float): Tie tolerance: values of g within this
            distance of 0 count as 0.
        exact_root (bool): Whether a point where g is within zero_tolerance of
            0 is the root itself, because g increases strictly through it, so
            that a root at 0, or a point of the secant, ends the search.
            Otherwise such a point may lie on a plateau where g is 0, and the
            search goes on to look for the end of that plateau.

    Returns:
        tuple[float, int]: The root, and the number of evaluations of g,
            including the start points, expansion points, midpoints and
            iterates.

    Raises:
        RuntimeWarning: g is within zero_tolerance of 0 at both 0 and upper
            (it does not respond); the bracket does not contain the root
            after _MAX_DOUBLINGS doublings in one direction; or the secant
            has not closed the bracket after _MAX_ITERATIONS iterations.
    """
    evaluations = 0

    def evaluate(c: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return g(c)

    lo, hi = 0.0, float(upper)
    points = []  # (c, g(c)) in evaluation order

    if not (guaranteed and method == "bisection"):
        # evaluate both ends, in this order, before any other test
        g_lo = evaluate(lo)
        g_hi = evaluate(hi)
        points = [(lo, g_lo), (hi, g_hi)]
        if abs(g_lo) <= zero_tolerance and abs(g_hi) <= zero_tolerance:
            message = (
                f"The metric does not respond to a step of {upper}; increase "
                "perturbation_size or use ExpectedUnservedEnergy"
            )
            LOG.error(message)
            raise RuntimeWarning(message)
        if exact_root and abs(g_lo) <= zero_tolerance:
            return lo, evaluations

        # expand downward while g(lo) is above 0, by upper, 2 * upper, ...
        doublings = 0
        while g_lo > zero_tolerance:
            if doublings == _MAX_DOUBLINGS:
                message = (
                    f"Could not bracket the root: g is still above 0 at {lo} "
                    f"after {_MAX_DOUBLINGS} doublings of the bracket downward"
                )
                LOG.error(message)
                raise RuntimeWarning(message)
            hi, g_hi = lo, g_lo
            lo -= upper * 2**doublings
            g_lo = evaluate(lo)
            points.append((lo, g_lo))
            doublings += 1

        # expand upward while g(hi) is below 0, doubling the upper end
        doublings = 0
        while g_hi < -zero_tolerance:
            if doublings == _MAX_DOUBLINGS:
                message = (
                    f"Could not bracket the root: g is still below 0 at {hi} "
                    f"after {_MAX_DOUBLINGS} doublings of the bracket upward"
                )
                LOG.error(message)
                raise RuntimeWarning(message)
            lo = hi
            hi *= 2
            g_hi = evaluate(hi)
            points.append((hi, g_hi))
            doublings += 1

        # an exact root, or a plateau of g = 0 that is not crossed
        if abs(g_hi) <= zero_tolerance:
            return hi, evaluations

    # both methods stop, with the midpoint, once the bracket cannot be split:
    # lo and hi are adjacent floating-point numbers, as tight as it can get
    if method == "bisection":
        while hi - lo > tolerance:
            midpoint = (lo + hi) / 2
            if not lo < midpoint < hi:
                break
            if evaluate(midpoint) <= zero_tolerance:
                lo = midpoint
            else:
                hi = midpoint
        return (lo + hi) / 2, evaluations

    # secant, with a bisection step whenever the previous step did not halve
    # the bracket
    bisect_next = False
    iterations = 0
    while hi - lo > tolerance:
        midpoint = (lo + hi) / 2
        if not lo < midpoint < hi:
            break
        if iterations == _MAX_ITERATIONS:
            message = (
                f"The root search did not converge in {_MAX_ITERATIONS} "
                f"iterations: the bracket [{lo}, {hi}] is wider than the "
                f"tolerance {tolerance}"
            )
            LOG.error(message)
            raise RuntimeWarning(message)
        iterations += 1

        (c1, g1), (c2, g2) = points[-2:]
        if bisect_next or g1 == g2:
            c = midpoint
        else:
            c = c2 - g2 * (c2 - c1) / (g2 - g1)
        if not lo < c < hi:
            c = midpoint
        elif abs(c - c2) < tolerance / 2:
            # the latest point is an end of the bracket: step away from it
            far_end = lo if c2 == hi else hi
            c = c2 + math.copysign(tolerance / 2, far_end - c2)

        g_c = evaluate(c)
        points.append((c, g_c))
        if exact_root and abs(g_c) <= zero_tolerance:
            return c, evaluations

        width = hi - lo
        if g_c <= zero_tolerance:
            lo = c
        else:
            hi = c
        bisect_next = hi - lo > width / 2
    return (lo + hi) / 2, evaluations


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
        The increment is computed multiply-first, as output * delta /
        nameplate, so that on integer data a shortfall the step clears
        exactly becomes exactly 0, as it does with the perfect resource.

        Args:
            baseline (_MarginalBaseline): Baseline of the evaluate() call.
            delta (float): Signed perturbation step in units of power.
            load (float, optional): Constant load added in every trial and
                hour, in units of power. Defaults to 0.

        Returns:
            float: Evaluated resource adequacy metric.
        """
        return self._responsive_risk(
            baseline.non_responsive_net
            + baseline.resource_output * delta / baseline.nameplate
            - load,
            baseline.storage_ids,
            delta,
            baseline.nameplate,
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
        delta: float = 0.0,
        nameplate: float = 1.0,
    ) -> float:
        """Dispatch the base system's responsive units onto a net hourly
        capacity matrix and return the resulting resource adequacy.

        Args:
            net_hourly_capacity_matrix (xr.DataArray): Net hourly capacity
                matrix after the non-responsive units. It is not modified.
            storage_ids (tuple[int, ...], optional): Ids of the storage units
                to enlarge. Defaults to none.
            delta (float, optional): Signed step by which those storage units
                are enlarged, in units of power. Defaults to 0.
            nameplate (float, optional): Total nameplate capacity to which
                the step is relative. Defaults to 1.

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
                    unit_dataset, list(storage_ids), delta, nameplate
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

        # adding 0.0 turns a factor of -0.0 (no effect, over a negative
        # denominator) into 0.0, and leaves every other value unchanged
        factor = (upper - lower) / denominator + 0.0
        LOG.info("Marginal reliability impact: " + str(factor))
        return factor


class MarginalEffectiveLoadCarryingCapability(_MarginalContributionMetric):
    """Class responsible for accrediting resources that are already in an
    energy system with their marginal effective load-carrying capability
    (ELCC): the constant load that the system can add while keeping its
    baseline resource adequacy when the resources are enlarged by a small
    step, relative to that step.

    Q. Zhang, F. Zhao, G. Stephen, C. Singh, L. Xie, "A Gradient-Based
    Capacity Accreditation Framework in Resource Adequacy: Formulation,
    Computation, and Practical Implications," arXiv:2601.22087, 2026.

    The resources keep their own sampled outages: every perturbed
    simulation of an evaluate() call reuses that call's samples (common
    random numbers). Enlarging the resources by a step scales their
    non-responsive output, and the charge rate, discharge rate, charge
    capacity and nameplate capacity of their storage units, by
    1 + step / nameplate capacity. As in EffectiveLoadCarryingCapability,
    the constant load is added after the non-responsive units, and the
    responsive units are re-dispatched in every simulation. Non-responsive
    units are not: units outside the resources keep their sampled output,
    and hydro units in the resources are not re-dispatched against each
    other's scaled output.

    With M(c) the resource adequacy when the resources are enlarged by the
    step and a constant load c is added, and M0 the baseline, the factor is
    the largest load c at which M(c) is no worse than M0 (the paper's
    Definition 1), divided by the step. The method keyword selects the
    search for that load, where the resolution is
    additional_demand_resolution_pct times the step and N_p is the number
    of simulations with the resources perturbed:

    - "bisection" (default): halves the search bracket until it is no
      wider than the resolution, and returns its midpoint. N_p = 7 at the
      default resolution when the bracket [0, step] is known to hold the
      load without evaluating its ends: with ExpectedUnservedEnergy, when
      the resources hold no storage and their output stays between 0 and
      their nameplate capacity, and when the only responsive units in the
      system are StorageUnit units. Otherwise both ends are evaluated
      first, and N_p is 2, plus the number of bracket expansions, plus
      ceil(log2(bracket width / resolution)).
    - "secant": the secant method of the paper (eq. 17), safeguarded as in
      Dekker's method by a bisection step whenever the secant step is
      undefined or would leave the bracket, or the previous step did not
      halve it. It evaluates both ends first. N_p depends on the data: it
      is often lower than with bisection (3 when the risk is linear in the
      load) and at most about twice the bisection's N_p plus 2.

    The search needs a metric whose risk never decreases as load is added,
    so resource_adequacy_metric must be exactly ExpectedUnservedEnergy,
    LossOfLoadHours or LossOfLoadDays. Other metrics, such as
    LossOfLoadFrequency, and subclasses of these three are refused;
    MarginalReliabilityImpact accepts any metric. LossOfLoadHours and
    LossOfLoadDays change in steps, so the risk can stay at its baseline
    level over a range of loads: if it is at that level at the upper end
    of the search bracket, that end is returned. The search does not cross
    the plateau, whose far end can lie well beyond the step.

    Unlike EffectiveLoadCarryingCapability, which returns power, evaluate()
    returns a dimensionless accreditation factor. The accredited capacity
    is the factor times the resources' nameplate capacity
    (resources.system_capacity). The factor is returned as computed, so it
    can be negative or above 1 for storage, for which the search bracket
    expands below 0 or above the step as needed. Also unlike
    EffectiveLoadCarryingCapability, additional_demand_resolution_pct is a
    fraction of the step, not of the nameplate capacity: at the default
    0.01, the factor is accurate to about 0.005.

    Each evaluate() call samples the system afresh (seed numpy's random
    number generator for reproducible results), and nothing is cached
    between calls: every call runs N_p perturbed simulations of the
    responsive units, plus 1 for the baseline M0.

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
        resource_adequacy_metric (type[ResourceAdequacyMetric]): Metric
            class to use throughout the calculation: ExpectedUnservedEnergy,
            LossOfLoadHours or LossOfLoadDays.
        perturbation_size (float | None, optional): Step in units of power.
            Defaults to None, for 1% of the nameplate capacity of the
            resources passed to evaluate().
        method (str, optional): "bisection" or "secant". Defaults to
            "bisection".

    Attributes:
        perturbation_count (int): Number of simulations with the resources
            perturbed (N_p) in the latest evaluate() call, also when it
            raised, or 0 before the first call. The baseline simulation is
            not counted: in total, a call evaluates resource adequacy
            N_p + 1 times.

    Raises:
        TypeError: energy_system is not an EnergySystem, or
            resource_adequacy_metric is not a ResourceAdequacyMetric class.
        ValueError: method is neither "bisection" nor "secant",
            perturbation_size is neither None nor a finite number greater
            than 0, or resource_adequacy_metric is not exactly
            ExpectedUnservedEnergy, LossOfLoadHours or LossOfLoadDays.
    """

    _methods = ("bisection", "secant")

    def __init__(
        self,
        energy_system: EnergySystem,
        simulation: ProbabilisticSimulation,
        resource_adequacy_metric: type[ResourceAdequacyMetric],
        perturbation_size: float | None = None,
        method: str = "bisection",
    ):
        _MarginalContributionMetric.__init__(
            self,
            energy_system,
            simulation,
            resource_adequacy_metric,
            perturbation_size,
            method,
        )
        # exact types, so that subclasses are refused
        if resource_adequacy_metric not in _ELCC_METRICS:
            message = (
                "resource_adequacy_metric must be exactly one of "
                + ", ".join(m.__name__ for m in _ELCC_METRICS)
                + f", got {resource_adequacy_metric.__name__}: the root "
                "search needs a metric whose risk is monotone, never "
                "decreasing as load is added (MarginalReliabilityImpact "
                "accepts any metric)"
            )
            LOG.error(message)
            raise ValueError(message)

    def evaluate(
        self,
        resources: EnergySystem,
        additional_demand_resolution_pct: float = 0.01,
    ) -> float:
        """Return the marginal effective load-carrying capability of
        resources that are already in the energy system.

        Args:
            resources (EnergySystem): Units of the base system to accredit
                together as one resource, e.g. a system built from some of
                its units, or energy_system.get_system_by_type(StorageUnit).
                Units are matched by exact unit type and id, and the base
                system's data is used.
            additional_demand_resolution_pct (float, optional): Resolution of
                the added load to find, as a fraction of the step (not of
                the nameplate capacity, unlike
                EffectiveLoadCarryingCapability), greater than 0 and less
                than 1. Defaults to 0.01 (1%), for a factor accurate to
                about 0.005. Resolutions far below 1e-12 gain nothing: the
                search stops at floating-point resolution, and below about
                1e-16 the secant can reach its iteration cap.

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
                greater than 0; additional_demand_resolution_pct is not
                greater than 0 and less than 1; or their output is negative
                in some trial and hour.
            RuntimeWarning: The base system has no risk; the metric does not
                respond to the step (the risk stays at its baseline level
                with no load and with a load of the step's size added); the
                search bracket does not hold the load after 10 doublings in
                one direction; or the secant has not converged after 100
                iterations.
        """
        self._perturbation_count = 0
        resource_ids, nameplate, step = self._resolve(resources)
        if not (
            isinstance(additional_demand_resolution_pct, numbers.Real)
            and 0 < additional_demand_resolution_pct < 1
        ):
            message = (
                "additional_demand_resolution_pct must be greater than 0 and "
                "less than 1 (a fraction of the perturbation step), got "
                f"{additional_demand_resolution_pct!r}"
            )
            LOG.error(message)
            raise ValueError(message)
        baseline = self._baseline(resource_ids, nameplate, step)
        eue = self._resource_adequacy_metric is ExpectedUnservedEnergy

        def g(load: float) -> float:
            # M(+step, load) - M0, counted as N_p also when the search raises
            self._perturbation_count += 1
            return self._risk(baseline, step, load) - baseline.risk

        # for EUE, g(0) <= 0 <= g(step) holds without evaluating either end
        # when the resources hold no storage (whose enlargement can raise
        # risk), their availability stays within [0, 1], and storage units,
        # whose dispatch is monotone, are the only responsive units
        output = baseline.resource_output.values
        guaranteed = (
            eue
            and not baseline.storage_ids
            and bool(((output >= 0) & (output <= nameplate)).all())
            and all(
                unit_type is StorageUnit
                for unit_type in self._original_energy_system.unit_datasets
                if unit_type in RESPONSIVE_UNIT_TYPES
            )
        )
        root, _ = _find_largest_root(
            g,
            upper=step,
            tolerance=additional_demand_resolution_pct * step,
            method=self._method,
            guaranteed=guaranteed,
            # counting metrics change in multiples of 1 / trials, so only
            # EUE needs a tolerance for ties
            zero_tolerance=1e-9 * baseline.risk if eue else 0.0,
            # EUE increases strictly with the load while it is above 0, so
            # a tie is the root itself rather than a point of a plateau
            exact_root=eue,
        )

        factor = root / step
        LOG.info(
            "Marginal effective load-carrying capability: " + str(factor)
        )
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
