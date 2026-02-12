wildfire_aspp
=============

Description
-----------

``wildfire_aspp`` is an explainable CNN model with an ASPP mechanism for next-day wildfire spread segmentation.
It is built via the PyHazards model registry.

Example usage
-------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_aspp",
       task="segmentation",
       in_channels=12,
   )

   x = torch.randn(2, 12, 64, 64)
   logits = model(x)
   print(logits.shape)  # (2, 1, 64, 64)

