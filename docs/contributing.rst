==========
Contribute
==========

Please contribute to the ASSET Lab Resource Adequacy Package!

Basic Contribution for New GitHub Users
----------------

Create a fork of `assetra <https://github.com/ijbd/assetra>`_ on GitHub.

Create a new issue describing the reason for your modification on the `issues <https://github.com/ijbd/assetra/issues>`_ page. Be sure to check if an open issue already exists!

Clone your fork locally from the command line.::

    git clone https://github.com/<your username>/assetra.git

Create a branch in your fork.::

    git switch -c <branch-name>

Modify files, then commit changes.::

    git status # show files modified
    git add <files> # space-separated list of files modified to be committed.
    git commit # will open a text editor for you to describe your changes.

If you haven't already, setup a GitHub token for your personal device.
- Create a token `here <https://github.com/settings/tokens>`_.
- Your token will have to be re-created after the expiration date you select.
- Ensure 'repo' and 'workflow' are checked.
- To store your token locally, run :code:`git config --global credential.helper store`. You will be prompted for your token the next time you commit. If needed, credentials can be discarded by running :code:`git config --global --unset credential.helper`.

Push local changes back to your fork on GitHub.::

    git push 

A Note on Poetry
----------------

The ASSET Lab Resource Adequacy Package uses `Poetry <https://python-poetry.org/>`_ to manage dependencies. 
From a contributors point of view, Poetry has two responsibilities:

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
