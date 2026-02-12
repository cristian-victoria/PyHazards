.. raw:: html

   <div style="margin: 30px 0; text-align: center;">
     <img src="_static/logo.png" alt="PyHazards Icon" style="max-width: 260px; height: auto;">
   </div>

.. image:: https://img.shields.io/pypi/v/pyhazards
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/ci.yml?branch=main
   :target: https://github.com/LabRAI/PyHazards/actions/workflows/ci.yml
   :alt: Build Status

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/LabRAI/PyHazards/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/downloads-check%20PyPI-blue
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub

----

**PyHazards** is a comprehensive Python framework for AI-powered hazard prediction and risk assessment. Built on PyTorch with a hazard-first design, the library provides a modular and extensible architecture for building, training, and deploying machine learning models to predict and analyze natural hazards and environmental risks.

**PyHazards is designed for:**

- **Hazard-First Architecture**: Unified dataset interface for tabular, temporal, and raster data
- **Simple, Extensible Models**: Ready-to-use MLP/CNN/temporal encoders with task heads
- **Trainer API**: Fit/evaluate/predict with optional mixed precision and multi-GPU (DDP) support
- **Metrics**: Classification, regression, and segmentation metrics out of the box
- **Extensibility**: Registries for datasets, models, transforms, and pipelines

**Quick Start Examples:**

Load one dataset:

.. code-block:: python

    from pyhazards.data.load_hydrograph_data import load_hydrograph_data

    data = load_hydrograph_data(
        era5_path="pyhazards/data/era5_subset",
        max_nodes=50,
    )
    print(data.feature_spec)
    print(data.label_spec)
    print(list(data.splits.keys()))  # ["train"]

Build one implemented model:

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

Core Components
---------------

**Datasets**
   PyHazards provides a unified dataset interface for tabular, temporal, and raster data, returning a ``DataBundle`` with splits and specs.

**Models**
   Extensible model architecture with MLP/CNN/temporal backbones and task heads for classification, regression, and segmentation.
   Easy to implement and register custom models via the model registry.

**Utilities**
   Helper functions for device management, seeding/logging, and metrics calculation.

How to Cite
-----------

If you use PyHazards in your research, please cite:

.. code-block:: bibtex

   @software{pyhazards2025,
     title={PyHazards: A Python Framework for AI-Powered Hazard Prediction},
     author={Cheng, Xueqi},
     year={2025},
     url={https://github.com/LabRAI/PyHazards}
   }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   pyhazards_datasets
   pyhazards_models
   pyhazards_engine
   pyhazards_metrics
   pyhazards_utils

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   cite
   references
   team


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
