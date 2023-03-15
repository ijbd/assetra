=========
Reference
=========

Overview
--------

Understanding the resource adequacy of an energy system, or its ability to meet demand, is a critical task for researchers studying energy transition, cross-sectoral system evolution, and energy system operation.
Researchers in the Advancing Sustainable Systems through low-impact Energy Technologies (ASSET) Lab and in institutions globally need usable research-grade tools to quantify resource adequacy as part of their multi-dimensional studies. 
The ASSET lab resource adequacy package (assetra) seeks to meet this need, with the overarching goal to quantify the resource adequacy of energy systems while being flexible, extensible, and easy-to-use for energy researchers.

The assetra object-oriented interface is shown in figure 1, and is easiest to interpret as a bottom-up model. *EnergyUnit* objects, such as demand centers and thermal generators, are added to *EnergySystemBuilder* objects. 

The *EnergySystemBuilder* compiles energy units into so-called unit datasets and instantiates *EnergySystem* objects. Unit datasets are `xarray <https://docs.xarray.dev/en/stable/index.html>`_ datasets on which higher-level objects operate (see [1]_ for additional justification). 

*EnergySystem* objects are added to *ProbabilisticSimulation* objects whose responsibility is to generate and store probabilistic net hourly capacity matrices for a large sample of Monte Carlo trials. The net hourly capacity matrix of a *ProbabilisticSimulation* is a two dimensional matrix whose elements represent the net system capacity in a given hour and Monte Carlo iteration. All of the resource adequacy metrics available in this package can be computed directly from a net hourly capacity matrix, so the *ProbabilisticSimulation* is stored as a member of the *ResourceAdequacyMetric*, which performs the relevant evaluation.

In addition to resource adequacy, the assetra package quantifies resource contribution of an addition to the energy system, specifically the addition's effective load-carrying capability (ELCC). Per the definition of ELCC, an *EffectiveLoadCarryingCapability* object computes the resource adequacy of a base *EnergySystem*, and then iteratively finds the constant load that can be met by additional resources at the same base resource adequacy. Because the computation of resource adequacy depends on both the simulation parameters and the selected resource adequacy metric, the *EffectiveLoadCarryingCapability* object is declared with a *ProbabilisticSimulation* object and *ResourceAdequacyMetric* type. 

.. figure:: _static/assetra-class-interface.drawio.png
   :scale: 50 %
   :alt: assetra class interface

   Figure 1: Class interface.

TODO custom units (responsive/non-responsive) good example is mttf outages
TODO temperature dependent outage rates

.. figure:: _static/assetra-inherited-types.drawio.png
   :scale: 50 %
   :alt: assetra inherited types

   Figure 2: Inherited types used in the assetra model.

Notes
-----
.. [1] Internally, we **try** to think of *EnergySystem* objects as immutable. There is no method to directly add, remove, or modify *EnergyUnit* objects to/from/in an *EnergySystem*. The reason for this is to make explicitly clear to users that higher level objects do not track the state of lower-level objects. For example, if a user wants to modify a system for which a probabilistic simulation has already been evaluated, it would be tedious to both recognize the system modification from the simulation object and preserve computation from the existing evaluation. Further, we want to make efficient use of data structures for larger simulations. For example, it is both time- and memory- efficient to operate on whole fleets of energy units via matrix operation rather than evaluating each unit individually. This also offers a pathway to future parallelization. On the other hand, it is important for users to modify systems, i.e. add or remove units at will, and it is convenient to think of energy units as individual conceptual objects (not as fleets). To summarize, the internal energy system model should be immutable and operate on fleets of energy units, while the external model should be modifiable and treat energy units as individual objects. The *EnergySystemBuilder* acts as a bridge between these two models.

Modules
-------
.. toctree::
    :maxdepth: 3

    assetra.units
    assetra.system
    assetra.simulation
    assetra.metrics
    assetra.contribution
