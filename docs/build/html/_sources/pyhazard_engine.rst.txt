Engine
===================

Summary
-------

The engine wraps training/evaluation/prediction with sensible defaults and optional distributed support.

Core modules
------------

- ``pyhazard.engine.trainer`` — ``Trainer`` class with ``fit``, ``evaluate``, ``predict``, checkpoint save.
- ``pyhazard.engine.distributed`` — strategy selection and config helpers.
- ``pyhazard.engine.inference`` — sliding-window inference placeholder for large rasters/grids.

Typical usage
-------------

.. code-block:: python

    import torch
    from pyhazard.engine import Trainer
    from pyhazard.metrics import ClassificationMetrics
    from pyhazard.models import build_model

    model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
    trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(data_bundle, optimizer=optimizer, loss_fn=loss_fn, max_epochs=10)
    results = trainer.evaluate(data_bundle, split="test")
    preds = trainer.predict(data_bundle, split="test")

Distributed and devices
-----------------------

- ``Trainer(strategy="auto")`` uses DDP when multiple GPUs are available; otherwise runs single-device.
- ``mixed_precision=True`` enables AMP when on CUDA.
- Device selection is handled via ``pyhazard.utils.hardware.auto_device`` by default.
