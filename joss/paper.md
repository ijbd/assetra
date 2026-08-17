---
title: '`assetra`: A Light-Weight Python Package for Resource Adequacy'
tags:
  - Python
  - energy systems
  - resource adequacy
  - effective load carrying capability
  - Xarray
authors:
  - name: Isaac Bromley-Dulfano
    orcid: 0000-0001-5868-6170
    affiliation: 1
  - name: Martha Vierra
    orcid: 0009-0000-9767-2978
    affiliation: "1, 2, 3"
  - name: Srihari Sundar
    affiliation: 1
    orcid: 0000-0002-0556-3967
  - name: Michael Craig
    corresponding: true
    orcid: 0000-0002-3031-5041
    affiliation: "1, 2, 4"
affiliations:
  - name: Center for Sustainable Systems, University of Michigan, Ann Arbor, MI, USA
    index: 1
  - name: School for Environment and Sustainability, University of Michigan, Ann Arbor, MI, USA
    index: 2
  - name: Department of Climate and Space Sciences and Engineering, University of Michigan, Ann Arbor, MI, USA
    index: 3
  - name: Department of Industrial and Operations Engineering, University of Michigan, Ann Arbor, MI, USA
    index: 4
date: XX September 2025
bibliography: paper.bib
---

# Summary

One of the imperatives of power system planners, operators, and regulators globally is to maximize power system reliability. As they modify existing practices for managing electrical grids in response to new technologies and climate change, reliability metrics benchmark and set targets for changing systems. Resource Adequacy (RA) refers to a set of reliability metrics that characterize the likelihood, frequency, and magnitude of “shortfall events”, or instances when demand exceeds available supply. RA analyses typically simulate the availability of generators across a range of operating conditions, and compare time series of available capacity against expected electrical demand. Researchers and practitioners need light-weight, flexible tools to incorporate RA within their analytical frameworks. The ASSET Lab RA package (`assetra`) is an easy-to-use and extensible Python package that offers a concise and intuitive interface for constructing representations of energy systems, running probabilistic simulations, and evaluating a number of common RA metrics. Additionally, `assetra` implements the effective load-carrying capability metric, which estimates the RA contributions of new resources to an energy system. With these features, `assetra` offers researchers and practitioners a tool for maintaining system reliability while advancing decarbonization and climate adaptation.


# Statement of Need and State of the Field 

As electrification, the transition to low-carbon energy sources, and changes in weather patterns from climate change occur simultaneously, understanding the changing RA of the electricity grid is vital [@hari23]. The `assetra` package contributes to this growing area of research by offering an easy-to-use and extensible Python package that offers a concise and intuitive interface for constructing representations of energy units and systems, running probabilistic simulations, and evaluating a number of common RA metrics. Our methodology reflects a tighter coupling between RA and meteorological modeling, and the need for tools that cater to interdisciplinary researchers [@craig22]. 

Existing open-source RA packages include the Probabilistic Resource Adequacy Suite (PRAS), ProGRESS, and GridPath, each of which addresses a different piece of the RA problem while leaving specific gaps for meteorologically-driven, interdisciplinary research. PRAS offers a sequential Monte Carlo simulation framework, including approximations for inter-regional transmission and energy storage, and its system model supports time-varying forced outage rates [@pras]. However, PRAS is written in Julia and is used as an importable library within Julia scripts, so integrating PRAS with the Python Xarray-based meteorological datasets common in climate-driven RA research requires an additional, non-trivial translation layer between languages. ProGRESS is written in Python and offers advanced modeling of energy storage devices, including charge/discharge dynamics and state-of-charge tracking, within its probabilistic simulation framework [@bera2025probabilistic]. Yet its input structure requires rigidly formatted CSV files for weather data and forced outage rates are specified as static per-generator values rather than a time-varying series, which limits its ability to accommodate large weather data files or model weather-driven outages. GridPath, also written in Python, embeds RA within a larger modeling framework that includes capacity expansion, production cost, and asset valuation [@mileva_2026]. GridPath's extensibility is substantial, but it is achieved through its SQLite-backed scenario database and Pyomo optimization framework, making customization difficult. 

These existing tools illustrate a common tradeoff between modeling capability and accessibility: substantial functionality paired with either a language barrier (PRAS), a rigid input schema (ProGRESS), or platform lock-in (GridPath), each of which raises the software knowledge required to adapt the tool to new weather data types or system configurations. `assetra` is designed to close these specific gaps rather than duplicate existing functionality: it is implemented natively in Python around Xarray, giving it direct compatibility with the Python meteorological and climate data ecosystem that PRAS cannot offer without a Julia-Python bridge; it accepts user-defined, arbitrarily-structured energy system and weather data through a flexible object-oriented interface rather than ProGRESS's fixed CSV schema; and it supports time-varying forced outage rates as a first-class feature for representing weather-driven generator unavailability. The result is a lightweight, extensible package that lets users customize their energy system setup and meteorological data pipelines through simple configuration, without requiring platform-specific database workflows or cross-language integration.

# Software Design

The key features which differentiate `assetra` from existing RA packages are the following:

- We define concise base classes to enable efficient development of custom technologies and RA metrics.
- We use Xarray data structures for all input and internal data structures.
- We provide an interface for time-varying forced outage rates, which, coupled with weather data, can be used to capture temperature-dependent forced outage rates.

In an evolving power system, quantifying RA, including tail risks and uncertainty, is vital [@epri24]. This process requires a highly interpretable methodology that enables researchers to analyze detailed statistics across numerous simulations. `assetra` stores simulation results for a researcher-specified sample of Monte Carlo trials in Xarray. Xarray organizes these results into data structures indexed by trial number and pandas datetime objects, thereby facilitating the interpretation of patterns in RA failures. Xarray was developed for use in the meteorological community, and allows for easy integration of climate data into `assetra` simulations [@hoyer2017xarray]. Its capabilities in lazy loading and efficient memory handling minimize memory overhead, enabling the processing of hundreds of simulations with hourly weather data. `assetra` also leverages Xarray's in-place operations to compute standard RA metrics for users, while providing researchers and developers with the flexibility to explore risks and uncertainties through more innovative approaches. 

RA modeling is often complex, computationally challenging, and inflexible [@esig20] [@esig24]. However, increasing levels of wind, solar, and storage technologies, along with evolving demand patterns, mean that RA models must capture the reliability contributions of a diverse array of resources [@esig20]. The `assetra` package employs a bottom-up approach to understanding RA, beginning with the individual `EnergyUnit` objects that constitute the system, as displayed in Figure 1. `assetra` `EnergyUnits` operate with time-varying capacity availability based on given weather profiles. Researchers can define generators in `assetra` with a nameplate capacity, an array of hourly maximum capacities, and a parallel array of hourly forced outage rates. Simple abstract base classes and class interfaces streamline data management complexity and enable customization to address specific research questions. The incorporated `StochasticUnit` and `StaticUnit` offer versatility to represent any generation technology, from data centers utilizing demand response programs to large wind farms. The `StorageUnit` and `HydroUnit` serve as heuristic-based units that respond to system net capacity, enabling researchers to explore how modifications to standard behaviors of these units could impact future reliability. Our object-oriented approach grants researchers the flexibility to customize specific aspects of their RA testing, such as adding new EnergyUnits or modifying probabilistic simulation methods, without needing to duplicate or alter the existing framework.

![Figure 1. Simplified class interface diagram illustrating data flow through the `assetra` package. ](assetra-class-interface.drawio.png)

The `assetra` package also provides built-in methodology for quantifying expected unserved energy, loss of load hours, loss of load days, and loss of load frequency during the simulation period, shown in Figure 2. `assetra` also offers a methodology for calculating the effective load-carrying capacity of potential investments on RA. `assetra` iteratively evaluates the amount of load a new resource can serve while maintaining the reliability level of a base system, by storing and re-using probabilistic simulation data. Efficiently calculating reliability contributions of potential investments enables researchers to broaden the scope of their analyses [@warp21]. The provided framework in `assetra` can also be customized to evaluate user-defined RA metrics or assess the reliability contributions of multiple, simultaneous investments. 

![Figure 2. Derived types used in the `assetra` model.](assetra-inherited-types.drawio.png)

# Research Impact Statement
By prioritizing simplicity and accessibility, the `assetra` package aims to redirect the focus of energy researchers to addressing policy and climate outcomes. Currently, researchers in the ASSET Lab are using `assetra` to model climate change impacts on reliability in the Western United States, incorporating aspects of RA into capacity expansion planning, and testing the RA benefit of building adaptation strategies in a regional context [@warp21] [@Christino2026]. In future versions of `assetra`, we hope to integrate multi-region transmission, parallelization, and additional outage simulators. By offering a flexible and lightweight object-oriented framework, `assetra` empowers researchers to model RA with ease and clarity, accommodating unique research needs through straightforward customization and integration. Whether addressing the challenges of novel methods or enhancing existing analysis frameworks, `assetra` provides a powerful tool for understanding the impacts of climate change and future investments on power system reliability. 

# Acknowledgements

We acknowledge Julian Florez, Reshmi Ghosh, and Pamela Wildstein, whose contributions to RA tools in the ASSET Lab inspired this work. This project was funded by NSF Grant No. 2142421, NSF Grant DGE 2241144, University of Michigan School for Environment and Sustainability, University of Michigan Undergraduate Research Opportunity Program, and the University of Michigan Institute for Energy Solutions. 

# AI Usage Disclosure 
AI tools were used in a limited manner to assist in debugging code during creation of this package. None of the code was written by AI, nor was AI used in the creation of package documentation or in the writing of this manuscript. 

# References 
