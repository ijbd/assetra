==========
Contribute
==========

Please contribute to the ASSET Lab Resource Adequacy Package! Refer to the guidelines below, intended for folks with wide range of open-source development experience.

Setup
-----

#. Install `Python 3 <https://www.python.org/downloads/>`_.

#. Install `https://git-scm.com/downloads`_.

#. Create a `GitHub <https://github.com/>`_ Account.

#. Fork `assetra <https://github.com/ijbd/assetra>`_ on GitHub. 
   
    * This creates your own version of the assetra repository stored at `https://github.com/<your username>/assetra.git`. You will make changes to your fork and submit them for review to be added to the main assetra repository.

#. Clone your Fork Locally.

    #. Navitage to the directory where you would like your clone to live.::

        git clone https://github.com/<your username>/assetra.git

#. Setup Poetry

    * Poetry is used to (1) manage virtual environments used to test development code, (2) maintain the :code:`pyproject.toml` file which defines package dependencies to be installed with :code:`assetra` by end-users, and (3) automate the process of building the :code:`assetra` package and publishing to pypi. When a contributor updates their environment (i.e. they install new or upgrade existing dependencies that need to be shipped with assetra), they should commit their modifications to the :code:`pyproject.toml` and :code:`poetry.ock` files. Other contributors should then run `poetry install` to remain in sync.

    #. Install pipx::

        python3 -m pip install pipx

    #. Install Poetry::

        pipx install poetry

    #. Create a Virtual Environment::

        python3 -m venv env

    #. Activate the Virtual Environment::

        source env/bin/activate # linux
        env/Scripts/activate # windows

    #. Install Dependencies::

        poetry install

    #. Test Installation::

        poetry run python -m unnittest
       
#. Setup GitHub Token
    
    #. Create a token `here <https://github.com/settings/tokens>`_.
        
        * Your will re-create your token after the expiration date you select.
        * Ensure 'repo' and 'workflow' are checked.
    
    #. Store your token locally.::
        
        git config --global credential.helper store 
   
    * You will be prompted for your token the next time you push changes. 
    * To unset credentials (e.g. after they expire), use :code:`echo "url=https://github.com" | git credential reject`.

#. Push local changes back to your fork on GitHub. Use your GitHub username and token if prompted for username and password.::

    git push 

Open a pull request to ijbd/assetra to submit your modifications for review.

Sync your fork to match ijbd/assetra from GitHub.





#. Modify files, then commit changes.::

    git status # show files modified
    git add <files> # space-separated list of files modified to be committed.
    git commit # will open a text editor for you to describe your changes.



#. Create a new issue describing the reason for your modification on the `issues <https://github.com/ijbd/assetra/issues>`_ page. Be sure to check if an open issue already exists!

   
   
Setup Poetry
----------------

The assetra package uses `Poetry <https://python-poetry.org/>`_ to manage dependencies. 
From a contributor's point of view, Poetry has two responsibilities:

1. Manage the virtual environment used to run and test code.
2. Update the 'pyproject.toml' file which defines dependencies for installing assetra.

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
