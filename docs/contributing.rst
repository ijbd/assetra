==========
Contribute
==========

Please contribute to the ASSET Lab Resource Adequacy Package! The guidelines below are intended for contributors with a wide range of open-source development experience.

Setup
-----

1. Install `Python 3 <https://www.python.org/downloads/>`_.

#. Install `Git <https://git-scm.com/downloads>`_.

#. Create a `GitHub <https://github.com/>`_ Account. Setup authentication via `SSH Key <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh>_` or `Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>_`.

#. Fork `:code:`assetra` <https://github.com/ijbd/assetra>`_ on GitHub. 
   
    * This creates your own version of the :code:`assetra` repository. You will make changes to your fork and submit them for review to be added to the main :code:`assetra` repository.

#. Clone your Fork Locally.

    #. Navitage to the directory where you would like your clone to live.::

        git clone https://github.com/<your username>/assetra.git

#. Setup Poetry

    * Poetry is used to (1) manage virtual environments used to test development code, (2) maintain the :code:`pyproject.toml` file which defines package dependencies to be installed with :code:`assetra` by end-users, and (3) automate the process of building the :code:`assetra` package and publishing to pypi. When a contributor updates their environment (i.e. they install new or upgrade existing dependencies that need to be shipped with :code:`assetra`), they should commit their modifications to the :code:`pyproject.toml` and :code:`poetry.lock` files. Other contributors should then run `poetry install` to remain in sync.

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
       
Contributor Workflow
--------------------

1. Identify an existing issue or open a new one in `GitHub <https://github.com/ijbd/assetra/issues>`.

#. Create a branch in your fork to work on your contributions.

#. Follow these guidelines for contributions:
'
    1. aa
    #. bb
    
#. Push local changes back to your fork on GitHub. Use your GitHub username and token if prompted for username and password.::

    git push 

Open a pull request to :code:`ijbd/assetra` to submit your modifications for review.

Sync your fork to match :code`ijbd/assetra` from GitHub.


Reviewer Workflow
-----------------


1. Modify files, then commit changes.::

    git status # show files modified
    git add <files> # space-separated list of files modified to be committed.
    git commit # will open a text editor for you to describe your changes.

    
#. Create a new issue describing the reason for your modification on the `issues <https://github.com/ijbd/assetra`/issues>`_ page. Be sure to check if an open issue already exists!

