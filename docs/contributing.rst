==========
Contribute
==========

Please contribute to the ASSET Lab Resource Adequacy Package! The guidelines below are intended for contributors with a wide range of open-source development experience.

Setup
-----

1. `Install Python 3 <https://www.python.org/downloads/>`_

#. `Install Git <https://git-scm.com/downloads>`_

#. `Create a GitHub Account <https://github.com/>`_

#. Setup GitHub Authentication. Either:

    a. `Setup GitHub SSH Keys <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh>`_
    
    b. `Setup Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_

#. Fork :code:`assetra` on `GitHub <https://github.com/ijbd/assetra>`_. 
   
    * This creates your own version of the :code:`assetra` repository. You will make changes to your fork and submit them for review to be added to the main :code:`assetra` repository.

#. Clone your Fork Locally.

    a. Navitage to the directory where you would like your clone to live.::

        git clone https://github.com/<your username>/assetra.git

#. Setup Poetry

    * Poetry is used to (1) manage virtual environments used to test development code, (2) maintain the :code:`pyproject.toml` file which defines package dependencies to be installed with :code:`assetra` by end-users, and (3) automate the process of building the :code:`assetra` package and publishing to pypi. When a contributor updates their environment (i.e. they install new or upgrade existing dependencies that need to be shipped with :code:`assetra`), they should commit their modifications to the :code:`pyproject.toml` and :code:`poetry.lock` files. Other contributors should then run :code:`poetry install` to remain in sync.

    a. Install pipx::

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

        poetry run python -m unittest
       
Contributor Workflow
--------------------

1. Identify an existing issue or open a new one in `GitHub <https://github.com/ijbd/assetra/issues>`_.

#. Create a branch in your fork to work on your contributions.

#. Modify files. Follow these guidelines when contributing:

    * For every function added or modified:
    
        a. Add at least one unit test.

    #. For every new EnergyUnit added:
     :code:`usage.rst`
     
        a. Add an example in the :code:`examples` folder. Include all data needed to run a working example, but be mindful of the file sizes.
        
        #. Update :code:`docs/reference.rst` to help others understand what you are modeling and what assumptions you make in your implementation.
    
#. Run all unit tests. Ensure they run successfully.
    
       poetry run python -m unittest
    
#. Commit changes::

    git status # show files modified
    git add <files> # space-separated list of files modified to be committed.
    git commit # will open a text editor for you to describe your changes.
    
#. Push local changes back to your fork on GitHub::

    git push 

#. Open a pull request to :code:`ijbd/assetra` to submit your modifications for review.

Reviewer Workflow
-----------------

1. Review code modifications.

#. Review new/modified unit tests.

#. Review new/modified examples.

#. Run all unit tests.

#. Run new/modified examples.

#. Request changes or accept the pull request.

#. Release the new code:

    a. Bump the code version in :code:`assetra/__init__.py` and :code:`pyproject.toml`. Commit these changes.

    #. Build the package::

        poetry build

    #. Publish package to pypi (requires pypi token)::

        poetry publish --build



