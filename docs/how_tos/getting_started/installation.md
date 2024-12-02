
```{eval-rst}
.. _installation:
```

# Installation

#. Install Python 3.11 or later.

#. Create a Python 3.11+ virtual environment. This example uses the ``venv`` module in the standard
library to create a virtual environment in your home directory. You may prefer a single
`python-envs` in your home directory instead of the current directory. You may also prefer ``conda``
or ``mamba``.

```{eval-rst}
.. code-block:: console

   $ python -m venv env
```

#. Activate the virtual environment.

```{eval-rst}
.. code-block:: console

   $ source env/bin/activate
```

Whenever you are done using chronify, you can deactivate the environment by running ``deactivate``.

#. Install the Python package `chronify`.

```{eval-rst}
.. code-block:: console

    $ pip install chronify
```
