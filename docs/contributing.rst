=====
Contribute
=====

Please feel free contribute to the ASSET Lab Resource Adequacy Package.

Using Poetry
------------

The ASSET Lab Resource Adequacy Package uses the Poetry dependency management tool. 
From a contributors point of view, Poetry is responsible for two tasks:

1. Managing the virtual environment used to run and test code.
2. Updating the 'pyproject.toml' file which documents dependencies for deployment.

Install pipx ::

    python -m pip install pipx

Installing Poetry ::

    pipx install poetry

Poetry recognizes existing virtual environments. To create a new environment ::

    python -m venv env

To activate the virtual environment ::

    source env/bin/activate

To install the defined dependencies ::

    poetry install

Poetry is used to synchronize contributors' environments. 
Whenever an environment change is made (i.e. packages are installed or updated), 
contributors should commit their modifications to the `pyproject.toml` and `poetry.lock` files.
Other contributors then run `poetry install` to remain in sync.
