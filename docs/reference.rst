=========
Reference
=========

Motivation
----------
Understanding the resource adequacy of an energy system, or its ability to meet demand, is a critical task for researchers studying energy transition, cross-sectoral system evolution, and energy system operation.
Researchers in the Advancing Sustainable Systems through low-impact Energy Technologies (ASSET) Lab and in institutions globally need usable research-grade tools to quantify resource adequacy as part of their multi-dimensional studies. 
The ASSET lab resource adequacy package (assetra) seeks to meet this need, with the overarching goal to quantify the resource adequacy of energy systems while being flexible, extensible, and easy-to-use for energy researchers.

Interface Overview
------------------
The assetra interface is object-oriented and best interpreted as a bottom-up model: individual resources are collected into an energy system, the system is simulated probabilistically, 
and the simulation is summarized by one or more resource adequacy metrics. Figure 1 shows the simplified class interface.

*EnergyUnits* are the building blocks of an `assetra` model. Each *EnergyUnit* represents a resource with an hourly capacity profile — a demand center, a thermal generator, a wind farm, a battery. 
Units are added one at a time to an *EnergySystemBuilder*, which is responsible for managing them and assembling them into a system.

*EnergySystems* are produced by calling *build()* on the builder. The resulting *EnergySystem* is an immutable collection of unit fleets rather than of individual units (see [1]_ for the motivation behind this distinction). 

*ProbabilisticSimulations* are where the Monte Carlo sampling happens. A *ProbabilisticSimulation* is instantiated with a start hour, an end hour, and a trial size, and is then assigned an 
*EnergySystem*. Calling *run()* dispatches every unit in the system across every trial and populates the net hourly capacity matrix: a two dimensional matrix of net system capacity
for each Monte Carlo iteration and each hour of the study period.

*ResourceAdequacyMetrics* turn the net capacity matrix produced by the *ProbabilisticSimulation* into report values for your simulation horizon, as exemplified in the metrics_example notebook. 
A *ResourceAdequacyMetric* is instantiated with a simulation object and evaluated with *evaluate()*. *ResourceAdequacyMetric* itself is an abstract base class; 
in practice users work with one of its concrete implementations:

.. list-table::
   :header-rows: 1
   :widths: 30 70
 
   * - Metric
     - Quantifies
   * - *ExpectedUnservedEnergy*
     - Total energy not served over the study horizon.
   * - *LossOfLoadHours*
     - Expected number of hours with a shortfall over the study horizon.
   * - *LossOfLoadDays*
     - Expected number of days with at least one shortfall hour over the study horizon.
   * - *LossOfLoadFrequency*
     - Expected number of distinct shortfall events over the study horizon. 



In addition to resource adequacy, the assetra package quantifies resource contribution of additional resources to an energy system, specifically with effective load-carrying capability (ELCC) metric. Per definition, an *EffectiveLoadCarryingCapability* object computes the resource adequacy of a base *EnergySystem*, and then iteratively finds the constant load that can be served by additional resources at the same base resource adequacy level. Resources that are already in the energy system can instead be accredited with a *MarginalEffectiveLoadCarryingCapability* or a *MarginalReliabilityImpact* (MRI) object, which measure how resource adequacy responds to a small increase in the size of those resources (see `Capacity Accreditation`_). Because the computation of resource adequacy depends on both the simulation parameters and the selected resource adequacy metric, each of these objects is composed of a base energy system, as well as a *ProbabilisticSimulation* object and *ResourceAdequacyMetric* type. 

.. figure:: _static/assetra-class-interface.drawio.png
   :scale: 50 %
   :alt: assetra class interface

   Figure 1: Simplified workflow diagram illustrating data flow through the `assetra` package.

Several core types in assetra are abstract base classes. Abstract interfaces allow for interchangeability and let users extend functionality — creating custom unit types or resource adequacy metrics 
without modifying the simulation framework. Figure 2 shows the abstract base classes and their derived types.

.. figure:: _static/assetra-inherited-types.drawio.png
   :scale: 50 %
   :alt: assetra derived types

   Figure 2: Derived types used in the assetra model.

Basic Workflow
--------------
A typical assetra study proceeds in two stages.
 
**Stage 1: Build the energy system**
 
1. Instantiate an *EnergySystemBuilder*.
2. Add a *DemandUnit* carrying the hourly demand profile for the system.
3. Add generating and storage capacity, choosing the unit type that matches each resource (see the table below).
4. Call *build()* to produce an *EnergySystem*.
 
**Stage 2: Simulate and evaluate**
 
1. Instantiate a *ProbabilisticSimulation* with a start hour, end hour, and Monte Carlo trial size.
2. Assign the energy system with *assign_energy_system()*.
3. Call *run()* to populate the net hourly capacity matrix.
4. Instantiate a resource adequacy metric with the simulation and call *evaluate()*.
 
Optionally, pass a base system, a simulation, and a metric type to an *EffectiveLoadCarryingCapability* object to quantify the contribution of additional resources, or to a *MarginalEffectiveLoadCarryingCapability* or *MarginalReliabilityImpact* object to accredit resources that are already in the system (see `Capacity Accreditation`_).

Choosing an Energy Unit Type
----------------------------
.. list-table::
   :header-rows: 1
   :widths: 22 48 30
 
   * - Unit type
     - Use for
     - Instantiated with
   * - *DemandUnit*
     - System demand. Treated identically to a static unit, but contributing negatively.
     - Hourly demand profile
   * - *StaticUnit*
     - Resources that always contribute their full profile, with no outage sampling.
     - Hourly capacity profile
   * - *HydroUnit*
     - Conventional hydro, either plant-level or aggregated regional hydro.
     - Monthly generation totals, nameplate capacity, hourly forced outage rates
   * - *StochasticUnit*
     - Thermal, solar, and wind generators subject to forced outages.
     - Hourly capacity profile, hourly forced outage rates
   * - *StorageUnit*
     - Battery and pumped hydro storage, dispatched with a greedy policy.
     - Charge/discharge capacity, energy capacity, efficiency

Preparing Input Data
--------------------
Energy units accept hourly profiles as `xarray <https://docs.xarray.dev/en/stable/index.html>`_ *DataArray* objects with a single :code:`time` dimension and datetime coordinates. 
This applies to every profile passed to a unit, including hourly capacity, hourly demand, and hourly forced outage rates. All units in a system must share a common time index, and the
simulation period requested from a *ProbabilisticSimulation* must fall within it. This may require some "data wrangling" in pre-processing as forced outage rates drawn from vulnerability curves
and demand data from utilities are unlikely to have matching time indices. The *assetra.utils* module provides a helper function to construct a time-indexed *DataArray* from a sequence of hourly values, 
which is useful for preparing input data coming from variety of sources.

Where a dataset already carries its own timestamps, it is usually clearest to construct the *DataArray* directly, so that the coordinates come from the source data rather than being assumed. 
The :code:`assetra.utils` module provides a shortcut for the other common case, in which the data is a plain sequence of hourly values with no index attached:

.. code-block:: python

   from assetra.utils import get_hourly_time_series_xr

   hourly_demand = get_hourly_time_series_xr(
       [100.0] * 8760, start_hour="2019-01-01 00:00:00"
   )

The values are assumed to be consecutive and evenly spaced at one-hour intervals, with the first value corresponding to :code:`start_hour`. 
The length of the input determines the length of the resulting time index, so a full non-leap year of hourly data is 8760 values. This helper is a convenience for formatting input data and is not part 
of the resource adequacy model. Units are indifferent to how their profiles were constructed, provided the dimension and coordinates are correct.

Dispatch Order
--------------
Units are not dispatched in the order they are added to the builder. They are dispatched by type, in a fixed order defined by the *RESPONSIVE_UNIT_TYPES* and *NONRESPONSIVE_UNIT_TYPES* variables 
in the *assetra.units* module [2]_. This order can be edited by declaring new lists with the desired order for your specific use case. 
The distinction is whether a unit's hourly capacity depends on full system conditions: responsive units dispatch last and alter output based on the net hourly capacity matrix.
 
The effective order is:
 
1. *DemandUnit* (non-responsive)
2. *StaticUnit* (non-responsive)
3. *HydroUnit* (non-responsive)
4. *StochasticUnit* (non-responsive)
5. *StorageUnit* (responsive)
 
The two units that are most impacted by dispatch order are the *HydroUnit* and *StorageUnit*. The *StorageUnit* is dispatched last, after every other unit type and both charges and discharges based on the net hourly capacity matrix. 
The *HydroUnit* is considered non-responsive, as it does not charge and discharge like the *StorageUnit*. However, hourly dispatch of the *HydroUnit* is dependent on the net demand in the system after the units 
preceding it in the *NONRESPONSIVE_UNIT_TYPES* list have been dispatched. The monthly generation profile for the *HydroUnit* is converted into an hourly capacity profile by distributing the 
monthly total across each hour of the month proportional to the net demand remaining after the generation from units listed before the Hydro Unit in NONRESPONSIVE_UNIT_TYPES are considered. 

Capacity Accreditation
----------------------
*EffectiveLoadCarryingCapability* quantifies the contribution of an addition to an energy system. Two marginal metrics, *MarginalEffectiveLoadCarryingCapability* (marginal ELCC) and *MarginalReliabilityImpact* (MRI), instead accredit resources that are already in the system, using those resources' own sampled outages. Both follow the gradient-based capacity accreditation framework of Zhang et al. [4]_: they enlarge the resources by a small step and measure how resource adequacy responds.

* *Marginal ELCC* is the largest constant load that the system can carry, with the resources enlarged by the step, while its resource adequacy stays no worse than the baseline (Definition 1 of [4]_), divided by the step.
* *MRI* is the change in resource adequacy when the resources are enlarged by the step, relative to the change when a perfect resource of the same size is added. A perfect resource adds the step in every trial and hour.

As shown in [4]_, the two coincide under *ExpectedUnservedEnergy* in the limit of a small step; at a finite step, and in computational cost, they can differ.

**Resources**

The resources to accredit are passed to *evaluate()* as an *EnergySystem* of units that are already in the base system: for example, one built with an *EnergySystemBuilder* from the same unit objects, or a whole fleet taken from the base system with *get_system_by_type()*. Units are matched to the base system by exact unit type and id, and the base system's data is used. *StaticUnit*, *StochasticUnit*, *HydroUnit* and *StorageUnit* units can be accredited, individually or as a group that is accredited as one resource; a *DemandUnit* cannot.

.. code-block:: python

   from assetra.contribution import (
       MarginalEffectiveLoadCarryingCapability,
       MarginalReliabilityImpact,
   )
   from assetra.metrics import ExpectedUnservedEnergy
   from assetra.units import StorageUnit

   # energy_system and simulation are built as in the basic workflow
   elcc = MarginalEffectiveLoadCarryingCapability(
       energy_system, simulation, ExpectedUnservedEnergy, method="secant"
   )
   mri = MarginalReliabilityImpact(
       energy_system, simulation, ExpectedUnservedEnergy, method="central"
   )

   wind = ...  # an EnergySystem of the wind units of energy_system (same ids)
   elcc.evaluate(wind)  # accreditation factor of the wind units
   mri.evaluate(energy_system.get_system_by_type(StorageUnit))  # of the whole storage fleet
   elcc.perturbation_count  # N_p of the latest evaluate() call

**Methods**

The :code:`method` keyword selects how the factor is computed. In the table, *M* is the resource adequacy with the resources enlarged by a signed step (reduced, for a negative step), *P* is the resource adequacy with a perfect resource of that signed size added instead, and *M0* is the baseline resource adequacy. The resolution of the marginal ELCC's load search is :code:`additional_demand_resolution_pct` (default 0.01) times the step.

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Metric and :code:`method`
     - Factor
     - Perturbed evaluations (N_p)
   * - *MRI*, :code:`"forward"` (default)
     - :code:`(M(+step) - M0) / (P(+step) - M0)`
     - 1
   * - *MRI*, :code:`"central"`
     - :code:`(M(+step) - M(-step)) / (P(+step) - P(-step))`
     - 2
   * - *Marginal ELCC*, :code:`"bisection"` (default)
     - The load of the marginal ELCC definition, divided by the step, found by halving a bracket of loads until it is no wider than the resolution.
     - 7 at the default resolution when the bracket :code:`[0, step]` is known to hold the load in advance (conditions in the class docstring). Otherwise both ends of the bracket are evaluated first, and the bracket is expanded if it does not hold the load.
   * - *Marginal ELCC*, :code:`"secant"`
     - The same load, found with the secant method of [4]_ (eq. 17), safeguarded by a bisection step whenever the secant step is undefined or would leave the bracket, or the previous step did not halve the bracket.
     - Depends on the data: often lower than bisection (3 when the risk is linear in the load), and at most about twice the bisection's N_p plus 2.

N_p, the number of resource adequacy evaluations with the resources perturbed, is the cost measure of [4]_ and is available as :code:`perturbation_count` after each *evaluate()* call. The baseline and perfect-resource evaluations are not counted: in total, a call evaluates resource adequacy N_p + 1 times for the marginal ELCC, and N_p + 2 (forward) or N_p + 3 (central) times for MRI. Only the responsive units are dispatched again in each evaluation; the non-responsive units are sampled once per call.

**Conventions**

* *Own outages.* The base system is sampled once per *evaluate()* call, and the resources' output is split out of those same draws, so every perturbed evaluation reuses the resources' own outages (common random numbers) instead of an independent copy of them. Each call samples afresh and nothing is cached between calls: seed numpy's random number generator for reproducible results.
* *Scaling.* Enlarging the resources by a step scales the output of their static, stochastic and hydro units by 1 + step / nameplate, where nameplate is the total nameplate capacity of the resources. For their storage units, :code:`charge_rate`, :code:`discharge_rate`, :code:`charge_capacity` and :code:`nameplate_capacity` are scaled by the same factor, while the initial state of charge (a fraction) and the efficiency are unchanged.
* *Responsive units.* Responsive units are dispatched again in every evaluation, including those with a perfect resource, so that they respond to the change.
* *Non-responsive units.* Non-responsive units are not dispatched again: units outside the resources keep their sampled output, including hydro units dispatched after the resources, and hydro units within the resources are not re-dispatched against each other's scaled output.
* *Added load.* The constant load of the marginal ELCC is added after the non-responsive units and before the responsive units are dispatched, as in *EffectiveLoadCarryingCapability*.
* *Plateaus.* *LossOfLoadHours* and *LossOfLoadDays* change in steps, so risk can stay at its baseline level over a range of loads. The marginal ELCC does not search across such a plateau: if risk is at its baseline level at the upper end of the search bracket, that end is returned, even when the plateau extends well beyond the step.

**Metrics and Step Size**

*MarginalReliabilityImpact* accepts any resource adequacy metric. *MarginalEffectiveLoadCarryingCapability* searches for a load, which needs a metric whose risk never decreases as load is added, so it accepts only *ExpectedUnservedEnergy*, *LossOfLoadHours* and *LossOfLoadDays*; other metrics, such as *LossOfLoadFrequency*, and subclasses of these three are refused with a :code:`ValueError` when the object is created.

By default the step is 1% of the total nameplate capacity of the resources passed to *evaluate()*. :code:`perturbation_size` sets an absolute step in units of power instead; for large fleets, an absolute step such as the 10 MW step of [4]_ (:code:`perturbation_size=10`, for a system in MW) can be used. With :code:`method="central"` the step cannot exceed the nameplate capacity of the resources, which the default always meets. *LossOfLoadHours* and *LossOfLoadDays* count shortfall hours and days, so they often do not respond to a step as small as 1%: use a larger step, or *ExpectedUnservedEnergy*. Where the factor is undefined, for example for a base system with no risk or a metric that does not respond to the step, the classes raise a :code:`RuntimeWarning` instead of returning a number. For the marginal ELCC, :code:`additional_demand_resolution_pct` is a fraction of the step, not of the nameplate capacity as in *EffectiveLoadCarryingCapability*; at the default 0.01, the factor is accurate to about 0.005.

**Interpreting the Result**

Unlike *EffectiveLoadCarryingCapability*, which returns power, *evaluate()* returns a dimensionless accreditation factor: the accredited capacity is the factor times :code:`resources.system_capacity`. The factor is returned as computed, so for storage it can be negative, because the greedy dispatch of storage can make risk rise when a storage unit is enlarged, or above 1, for example when the discharge rate exceeds the nameplate capacity.

Assumptions
-----------

**Static Units**
 - Static units are instantiated with hourly capacity profiles.
 - Static units always contribute their full hourly capacity.
 - Demand profiles, instantiated as `DemandUnit` objects, are treated identically to static units but with a negative contribution.

**Stochastic Units**
 - Stochastic units are instantiated with hourly capacity profiles and hourly forced outage rates.
 - Stochastic unit outages are sampled independently in each hour.
 - Stochastic units contribute zero capacity in hours where unit outages occur, otherwise they contribute their full hourly capacity.
 - Stochastic units are well-suited to model thermal, solar, and wind generators.

**Storage Units**
 - Storage units are dispatched with a greedy policy to minimize expected unserved energy. When net capacity exceeds demand, storage units charge. When system demand exceeds capacity, storage units discharge. Both charging and discharging are limited by the rated power capacity of the unit.
 - Storage units are dispatched sequentially according to the order in which they are added to the system.
 - Storage unit efficiency deratings are applied in equal part on charge and discharge. 
 - Storage units are initialized with full state of charge unless initial soc is set lower. 
 - Storage units are well-suited to model battery and pumped hydro storage [3]_.

**ProbabilisticSimulation**
 - Probabilistic simulations dispatch unit datasets in order of type. The established order is *DemandUnit*, *StaticUnit*, *StochasticUnit*, *StorageUnit* [2]_.

**Hydro Unit**
   (NOTE: only works for 12 months at a time right now, cannot run for longer time blocks)  
 - Hydro units borrow much from the stochastic unit but are initiated with monthly generation estimates.
 - Monthly generation profiles are intended to be an estimate of total generation for a unit in every month of a 12 month cycle, based on EIA annual collection of  monthly generation reports from operational hydropower units. 
 - The Hydro Unit turns the monthly generation profile into an hourly capacity profile by distributing the monthly total across each hour of the month proportional to the net demand remaining after the generation from units listed before the Hydro Unit in NONRESPONSIVE_UNIT_TYPES are considered.   
 - Hydro units are instantiated with hourly forced outage rates and nameplate capacities, and the net-demand-proportional capacity in each hour is limited by the nameplate capacity of the unit. 
 - Hydro unit outages are sampled independently in each hour.
 - Hydro units contribute zero capacity in hours where unit outages occur, otherwise they contribute their full hourly capacity as calculated from the proportional assignment of the monthly generation across all hours of the month.
 - Hydro units can be used to model individual hydropower plants or regional hydropower. For larger systems, modeling a single regional hydropower unit that accounts for all generation potential in the area will significantly reduce program run time as opposed to modeling every generator separately. 

Notes
-----
.. [1] Internally, we **try** to think of *EnergySystem* objects as immutable. There is no method to directly add, remove, or modify *EnergyUnit* objects to/from/in an *EnergySystem*. The reason for this is to make explicitly clear to users that higher level objects do not track the state of lower-level objects. For example, if a user wants to modify a system for which a probabilistic simulation has already been evaluated, it would be tedious to both recognize the system modification from the simulation object and preserve computation from the existing evaluation. Further, we want to make efficient use of data structures for larger simulations. For example, it is both time- and memory- efficient to operate on whole fleets of energy units via matrix operation rather than evaluating each unit individually. This also offers a straightforward path to future parallelization. On the other hand, it is important for users to modify systems, i.e. add or remove units at will, and it is convenient to think of energy units as individual conceptual objects (not as fleets). To summarize, the internal energy system model should be immutable and operate on fleets of energy units, while the external model should be modifiable and treat energy units as individual objects. The *EnergySystemBuilder* acts as a bridge between these two models, by initializing the *EnergySystem* with a list of `xarray <https://docs.xarray.dev/en/stable/index.html>`_ datasets representing immutable fleets.
.. [2] The dispatch order of unit datasets in probabilistic simulations is defined by two variables in the *assetra.units* module, specifically *RESPONSIVE_UNIT_TYPES* and *NONRESPONSIVE_UNIT_TYPES*. These two variables are lists which both define valid energy unit types and distinguish the order of unit dispatch. The responsive/non-responsive nomenclature refers to whether the hourly capacity of units of a given type depend on system conditions. For example, *StaticUnit* and *StochasticUnit* qualify as non-responsive because their probabilistic hourly capacities do not depend on the net hourly capacity matrix. *StorageUnit* on the other hand qualifies as a responsive type. Dispatch order follows the combined list *(NONRESPONSIVE_UNIT_TYPES + RESPONSIVE_UNIT_TYPES)*.
.. [3] Partial outages instead of full outages can be achieved by varying the hourly capacity factor of a generation unit. This functionality is specifically useful for wind and solar farms that may operate at only 25-50% capacity during certain time periods. 
.. [4] Q\. Zhang, F. Zhao, G. Stephen, C. Singh, L. Xie, "A Gradient-Based Capacity Accreditation Framework in Resource Adequacy: Formulation, Computation, and Practical Implications," `arXiv:2601.22087 <https://arxiv.org/abs/2601.22087>`_, 2026.


