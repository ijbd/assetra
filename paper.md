 ---
title: 'ASSETRA: A light-weight Python package for resource adequacy'
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
  - name: Srihari Sundar
    affiliation: 2
  - name: Michael Craig
    corresponding: true #TODO
    affiliation: "3, 4"
affiliations:
 - name: Research Assistant, School for Environment and Sustainability, University of Michigan, USA
   index: 1
 - name: PhD Candidate, School for Environment and Sustainability, University of Michigan, USA
   index: 2
 - name: Assistant Professor, School for Environment and Sustainability, University of Michigan, USA
   index: 3
 - name: Assistant Professor, Industrial and Operations Engineering, University of Michigan, USA
   index: 4
date: 31 March 2023
bibliography: paper.bib
---

# Summary

Understanding the resource adequacy of an energy system, or its ability to meet 
demand across a range of uncertain operating conditions, is a critical task for
research groups studying energy transition.
Researchers in the Advancing Sustainable Systems through low-impact Energy 
Technologies (ASSET) Lab and in institutions globally need usable research-
grade tools to quantify resource adequacy as part of their multi-dimensional
studies. The ASSET lab resource adequacy package (`assetra`) seeks to meet this 
need, with the overarching goal of quantifying the resource adequacy of energy 
systems while being flexible, extensible, and easy-to-use for researchers. 
In addition to providing a Monte Carlo simulation framework to quantify several
common resource adequacy metrics, `assetra` also implements the effective
load-carrying capability metric, which researchers can use to estimate the 
resource adequacy contributions of a new resource to existing energy systems.

# Statement of need

The `assetra` package is an xarray-based Python package for resource adequacy, 
prioritizing simplicity and ease-of-use. The object-oriented interface offers 
an intuitive bottom-up model of energy systems as a collection of energy units.
Building on xarray means that energy systems and simulation artifacts are 
stored in interpretable data structures, an unmistakable convenience for 
developers and users alike. The development of `assetra` stems from a 2021 
resource adequacy study by researchers in ASSET lab [@warp:2021]. The package 
is primarily being developed for two classes of studies: those which require 
novel methods unaccounted for in existing resource adequacy models, and those 
which are not primarily concerned with resource adequacy but wish to include 
resource adequacy as part of their multi-dimensional analyses. Consider two 
potential applications. First, suppose a research group needs to integrate 
demand response into their adequacy study (a topical application for the coming
decade). Though demand response is not implemented in the `assetra` package, 
researchers can define custom behavior for demand response by deriving from the 
abstract `EnergyUnit` class, requiring the implementation of only three 
functions. As another example, suppose a research group has successfully 
modeled future hourly load predictions under likely scenarios of climate 
change. They want to understand how climate change affects the resource 
adequacy contributions of near-term solar and wind investments. Their 
application does not require extension of `assetra`, but benefits primarily 
from its consice interface. Analyses of both flavors are being undertaken by 
researchers in ASSET Lab [TODO] and elsewhere [TODO]. With the joint 
developments of energy transition and climate change adaptation, `assetra` 
hopes to ease the mind of energy researchers, allowing them to focus less on 
Python and more on policy/climate outcomes.

TODO add to bib WARP
TODO cite/add to bib Reshmi
TODO cite/add to bib Hari
TODO cite/add to bib papers citing WARP for methods

# Acknowledgements

We acknowledge Julian Florez, Reshmi Ghosh, and Pamela Wildstein, whose contributions to resource adequacy tools in the ASSET Lab inspired this work. This project was funded by #TODO
