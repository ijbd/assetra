=======
ASSETRA
=======

.. image:: https://img.shields.io/pypi/v/assetra.svg
        :target: https://pypi.python.org/pypi/assetra

.. image:: https://readthedocs.org/projects/assetra/badge/?version=latest
        :target: https://assetra.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


The ASSET Lab Resource adequacy package (assetra) is a light-weight, open-source energy system resource adequacy package maintained by the University of Michigan ASSET Lab.


* Free software: MIT license
* Documentation: https://assetra.readthedocs.io.


Features
--------
* Probabilistic Monte Carlo state-sampling simulation framework, supporting:
        * Time-varying forced outage rates in thermal units
        * Sequential storage unit dispatch
        * User-defined energy unit types
* Resource adequacy calculation:
        * Expected unserved energy (EUE)
        * Loss of load hours (LOLH)
        * Loss of load days (LOLD)
* Resource contribution calculation:
        * Effective load-carrying capability (ELCC)
* Object-oriented interface to manage energy units within energy systems
* Internal computation stored in `xarray <https://docs.xarray.dev/en/stable/index.html>`_ datasets

Future Work
-----------
* Regional interchange and transmission
* Parallelized computation

Credits
-------
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
