.. raw:: html

   <div style="margin-top: 50px; text-align: center;">
     <img src="_static/logo.png" alt="PyHazard Icon" style="width: 600px; height: auto;">
   </div>

.. image:: https://img.shields.io/pypi/v/PyHazard
   :target: https://pypi.org/project/PyHazard
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazard/docs.yml
   :target: https://github.com/LabRAI/PyHazard/actions
   :alt: Build Status

.. image:: https://img.shields.io/github/license/LabRAI/PyHazard.svg
   :target: https://github.com/LabRAI/PyHazard/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/dm/pyhazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyHazard
   :alt: GitHub

----

**PyHazard** is a comprehensive Python framework for AI-powered hazard prediction and risk assessment. Built on PyTorch with a hazard-first design, the library provides a modular and extensible architecture for building, training, and deploying machine learning models to predict and analyze natural hazards and environmental risks.

**PyHazard is designed for:**

- **Hazard-First Architecture**: Unified dataset interface for tabular, temporal, and raster data
- **Simple, Extensible Models**: Ready-to-use MLP/CNN/temporal encoders with task heads
- **Trainer API**: Fit/evaluate/predict with optional mixed precision and multi-GPU (DDP) support
- **Metrics**: Classification, regression, and segmentation metrics out of the box
- **Extensibility**: Registries for datasets, models, transforms, and pipelines

**Quick Start Example:**

Basic Usage Example (toy dataset):

.. code-block:: python

    import torch
    from pyhazard.datasets import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
    from pyhazard.models import build_model
    from pyhazard.engine import Trainer
    from pyhazard.metrics import ClassificationMetrics

    class ToyHazard(Dataset):
        def _load(self):
            x = torch.randn(500, 16)
            y = torch.randint(0, 2, (500,))
            splits = {
                "train": DataSplit(x[:350], y[:350]),
                "val": DataSplit(x[350:425], y[350:425]),
                "test": DataSplit(x[425:], y[425:]),
            }
            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(input_dim=16, description="toy features"),
                label_spec=LabelSpec(num_targets=2, task_type="classification"),
            )

    data = ToyHazard().load()
    model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
    trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(data, optimizer=optimizer, loss_fn=loss_fn, max_epochs=5)
    results = trainer.evaluate(data, split="test")
    print(results)

Core Components
---------------

**Datasets**
   PyHazard provides a unified dataset interface for tabular, temporal, and raster data, returning a ``DataBundle`` with splits and specs.

**Models**
   Extensible model architecture with MLP/CNN/temporal backbones and task heads for classification, regression, and segmentation.
   Easy to implement and register custom models via the model registry.

**Utilities**
   Helper functions for device management, seeding/logging, and metrics calculation.

How to Cite
-----------

If you use PyHazard in your research, please cite:

.. code-block:: bibtex

   @software{pyhazard2025,
     title={PyHazard: A Python Framework for AI-Powered Hazard Prediction},
     author={Cheng, Xueqi},
     year={2025},
     url={https://github.com/LabRAI/PyHazard}
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

   pyhazard_datasets
   pyhazard_utils

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
