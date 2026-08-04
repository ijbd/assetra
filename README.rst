=======
ASSETRA
=======

.. image:: https://img.shields.io/pypi/v/assetra.svg
        :target: https://pypi.python.org/pypi/assetra

.. image:: https://readthedocs.org/projects/assetra/badge/?version=latest
        :target: https://assetra.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://github.com/ijbd/assetra/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/ijbd/assetra/actions/workflows/tests.yml
    :alt: Test status
    
.. image:: https://raw.githubusercontent.com/ijbd/assetra/main/.github/coverage.svg
    :target: https://github.com/ijbd/assetra/actions/workflows/tests.yml
    :alt: Test coverage
### Platform & Python Version Compatibility

| Operating System | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
| :--- | :---: | :---: | :---: | :---: |
| **Ubuntu** | ![Ubuntu 3.10](https://img.shields.io/badge/Python_3.10-passing-brightgreen?style=flat-square&logo=python&logoColor=white) | ![Ubuntu 3.11](https://img.shields.io/badge/Python_3.11-passing-brightgreen?style=flat-square&logo=python&logoColor=white) | ![Ubuntu 3.12](https://img.shields.io/badge/Python_3.12-passing-brightgreen?style=flat-square&logo=python&logoColor=white) | ![Ubuntu 3.13](https://img.shields.io/badge/Python_3.13-failing-red?style=flat-square&logo=python&logoColor=white) |
| **Windows** | ![Windows 3.10](https://img.shields.io/badge/Python_3.10-passing-brightgreen?style=flat-square&logo=python&logoColor=white) | ![Windows 3.11](https://img.shields.io/badge/Python_3.11-passing-brightgreen?style=flat-square&logo=python&logoColor=white) | ![Windows 3.12](https://img.shields.io/badge/Python_3.12-passing-brightgreen?style=flat-square&logo=python&logoColor=white) | ![Windows 3.13](https://img.shields.io/badge/Python_3.13-failing-red?style=flat-square&logo=python&logoColor=white) |

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
        * Loss of load frequency (LOLF)
* Resource contribution calculation:
        * Effective load-carrying capability (ELCC)
* Object-oriented interface to manage energy units within energy systems
* Internal computation stored in `xarray <https://docs.xarray.dev/en/stable/index.html>`_ datasets

Credits
-------
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
