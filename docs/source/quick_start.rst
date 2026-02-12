Quick Start
=================
This guide will help you get started with PyHazards quickly using the hazard-first API.

Basic Usage
-----------

How to load one dataset (real ERA5 subset for flood modeling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhazards.data.load_hydrograph_data import load_hydrograph_data

    data = load_hydrograph_data(
        era5_path="pyhazards/data/era5_subset",
        max_nodes=50,
    )

    print(data.feature_spec)
    print(data.label_spec)
    print(list(data.splits.keys()))  # ["train"]

How to build one implemented model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )
    print(type(model).__name__)

GPU Support
-----------

PyHazards automatically detects CUDA availability. To explicitly set the device:

**Using Environment Variable:**

.. code-block:: bash

    export PYHAZARDS_DEVICE=cuda:0

**Using Python API:**

.. code-block:: python

    from pyhazards.utils import set_device

    # Set to use CUDA device 0
    set_device("cuda:0")

    # Or use CPU
    set_device("cpu")

Next Steps
----------

For more detailed documentation, please refer to:

- :doc:`pyhazards_datasets` - Dataset interface and registration
- :doc:`pyhazards_utils` - Utility functions and helpers
- :doc:`implementation` - Guide for implementing custom datasets and models
