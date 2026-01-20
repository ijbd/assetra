---
title: '`assetra`: A Light-Weight Python Package for Resource Adequacy'
tags:
  - Python
  - energy systems
  - resource adequacy
  - effective load carrying capability
  - xarray
authors:
  - name: Isaac Bromley-Dulfano
    orcid: 0000-0001-5868-6170
    affiliation: 1
  - name: Martha Christino
    orcid: 0009-0000-9767-2978
    affiliation: "1, 2, 3"
  - name: Michael Craig
    corresponding: true
    orcid: 0000-0002-3031-5041
    affiliation: "1, 2, 4"
  - name: Srihari Sundar
    affiliation: 1
    orcid: 0000-0002-0556-3967
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

One of the imperatives of power system planners, operators, and regulators globally is to maximize power system reliability. As they modify existing practices for managing electrical grids in response to new technologies and climate change, reliability metrics benchmark and set targets for changing systems. Resource Adequacy (RA) refers to a set of reliability metrics that characterize the likelihood, frequency, and magnitude of “shortfall events”, or instances when demand exceeds available supply. RA analyses typically simulate the availability of generators across a range of operating conditions, and compare time series of available capacity against expected electrical demand. Researchers and practitioners need light-weight, flexible tools to incorporate RA within their analytical frameworks. The ASSET Lab resource adequacy package (`assetra`) is an easy-to-use and extensible Python package that offers a concise and intuitive interface for constructing representations of energy systems, running probabilistic simulations, and evaluating a number of common resource adequacy metrics. Additionally, `assetra` implements the effective load-carrying capability metric, which estimates the resource adequacy contributions of new resources to an energy system. With these features, `assetra` offers researchers and practitioners a tool for maintaining system reliability while advancing decarbonization and climate adaptation.


# Statement of Need

As electrification, the transition to low-carbon energy sources, and changes in weather patterns from climate change occur simultaneously, understanding the changing resource adequacy of the electricity grid is vital [@hari23]. The `assetra` package contributes to this growing area of research by offering an easy-to-use and extensible Python package that offers a concise and intuitive interface for constructing representations of energy units and systems, running probabilistic simulations, and evaluating a number of common resource adequacy metrics. Our methodology reflects a tighter coupling between resource adequacy and meteorological modeling, and the need for tools that cater to interdisciplinary researchers [@craig22]. Existing open-source RA packages include the Probabilistic Resource Adequacy Suite (PRAS), ProGRESS, and GridPath. PRAS, written in Julia, offers a sequential Monte Carlo simulation framework that includes approximations for inter-regional transmission and energy storage [@pras]. ProGRESS, written in Python, offers advanced modeling of energy storage devices within its probabilistic simulation framework [@progress]. GridPath, also written in Python, embeds RA within a larger modeling framework that includes RA, capacity expansion, production cost, and asset valuation [@gridpath]. 

# Software Design

The key features which differentiate `assetra` from existing RA packages are the following:

- We define concise base classes to enable efficient development of custom technologies and resource adequacy metrics.
- We use Xarray data structures for all input and internal data structures.
- We provide an interface for time-varying forced outage rates, which, coupled with weather data, can be used to capture temperature-dependent forced outage rates.

In an evolving power system, quantifying resource adequacy, including tail risks and uncertainty, is vital [@epri24]. This process requires a highly interpretable methodology that enables researchers to analyze detailed statistics across numerous simulations. `assetra` stores simulation results for a researcher-specified sample of Monte Carlo trials in Xarray. Xarray organizes these results into data structures indexed by trial number and pandas datetime objects, thereby facilitating the interpretation of patterns in resource adequacy failures. Xarray was developed for use in the meteorological community, and allows for easy integration of climate data into `assetra` simulations [@xarray]. Its capabilities in lazy loading and efficient memory handling minimize memory overhead, enabling the processing of hundreds of simulations with hourly weather data. `assetra` also leverages Xarray's in-place operations to compute standard resource adequacy metrics for basic users, while providing researchers and developers with the flexibility to explore risks and uncertainties through more innovative approaches. 

Resource adequacy modeling is often complex, computationally challenging, and lacks flexibility [@esig20] [@esig24]. However, increasing levels of wind, solar, and storage technologies, along with evolving demand patterns, mean that our resource adequacy models must capture the reliability contributions of a diverse array of resources [@esig20]. The `assetra` package employs a bottom-up approach to understanding resource adequacy, beginning with the individual `EnergyUnit` objects that constitute the system.  `assetra` `EnergyUnits` operate with time-varying capacity availability based on given weather profiles. Researchers can define generators in `assetra` with a nameplate capacity, an array of hourly maximum capacities, and a parallel array of hourly forced outage rates. Simple abstract base classes and class interfaces streamline data management complexity and enable customization to address specific research questions. The incorporated `StochasticUnit` and `StaticUnit` offer versatility to represent any generation technology, from data centers utilizing demand response programs to large wind farms. The `StorageUnit` and `HydroUnit` serve as heuristic-based units that respond to system net capacity, enabling researchers to explore how modifications to standard behaviors of these units could impact future reliability. Our object-oriented approach grants researchers the flexibility to customize specific aspects of their resource adequacy testing, such as adding new EnergyUnits or modifying probabilistic simulation methods, without needing to duplicate or alter the existing framework.

The `assetra` package also provides a method for quantifying the effective load-carrying capacity of potential investments on resource adequacy. `assetra` iteratively evaluates the amount of load a new resource can serve while maintaining the reliability level of a base system, by storing and re-using probabilistic simulation data. Efficiently calculating reliability contributions of potential investments enables researchers to broaden the scope of their analyses [@warp21]. The provided framework can be customized to evaluate user-defined resource adequacy metrics or assess the reliability contributions of multiple, simultaneous investments.

# Research Impact Statement
By prioritizing simplicity and accessibility, the `assetra` package aims to redirect the focus of energy researchers to addressing policy and climate outcomes. Currently, researchers in the ASSET Lab are using `assetra` to model climate change impacts on reliability in the Western United States, incorporating aspects of resource adequacy into capacity expansion planning, and testing the resource adequacy benefit of building adaptation strategies in a regional context [@warp21] [@Christino2026]. In future versions of `assetra`, we hope to integrate multi-region transmission, parallelization, and additional outage simulators. By offering a flexible and lightweight object-oriented framework, `assetra` empowers researchers to model resource adequacy with ease and clarity, accommodating unique research needs through straight-forward customization and integration. Whether addressing the challenges of novel methods or enhancing existing analysis frameworks, `assetra` provides a powerful tool for understanding the impacts of climate change and future investments on power system reliability. 

# Acknowledgements

We acknowledge Julian Florez, Reshmi Ghosh, and Pamela Wildstein, whose contributions to resource adequacy tools in the ASSET Lab inspired this work. This project was funded by NSF Grant No. 2142421, NSF Grant DGE 2241144, University of Michigan School for Environment and Sustainability, University of Michigan Undergradute Research Oppurtunity Program, and the University of Michigan Institue for Energy Solutions. 

# AI Usage Disclosure 
AI tools were used in a limited manner to assist in debugging code during creation of this package. None of the code was written by AI, nor was AI used in the creation of package documentation or in the writing this manuscript. 
