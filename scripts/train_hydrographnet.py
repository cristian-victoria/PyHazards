#example
#!/usr/bin/env python3

import torch

from pyhazards.datasets import (
    GraphTemporalDataset,
    graph_collate,
    DataBundle,
    DataSplit,
    FeatureSpec,
    LabelSpec,
)
from pyhazards.engine import Trainer
from pyhazards.models import build_model
from pyhazards.models.hydrographnet import HydroGraphLoss


def main():
    # -----------------------------
    # Dummy example data (REPLACE later)
    # -----------------------------
    samples = 64
    past_days = 1          # HydroGraphNet is one-step for now
    num_nodes = 10
    node_feats = 6
    edge_feats = 3

    # Node features: (samples, past_days, N, F)
    x = torch.randn(samples, past_days, num_nodes, node_feats)

    # Targets: (samples, N, out_dim)
    y = torch.randn(samples, num_nodes, 1)

    # Adjacency (shared graph)
    adjacency = torch.eye(num_nodes)

    # -----------------------------
    # Dataset
    # -----------------------------
    train_ds = GraphTemporalDataset(
        x[:48],
        y[:48],
        adjacency=adjacency,
    )

    val_ds = GraphTemporalDataset(
        x[48:],
        y[48:],
        adjacency=adjacency,
    )

    bundle = DataBundle(
        splits={
            "train": DataSplit(inputs=train_ds, targets=None),
            "val": DataSplit(inputs=val_ds, targets=None),
        },
        feature_spec=FeatureSpec(
            input_dim=node_feats,
            extra={"past_days": past_days, "num_nodes": num_nodes},
        ),
        label_spec=LabelSpec(
            num_targets=1,
            task_type="regression",
        ),
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=node_feats,
        edge_in_dim=edge_feats,
        out_dim=1,
    )

    # -----------------------------
    # Optimizer + Loss
    # -----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = HydroGraphLoss()

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(model=model)

    trainer.fit(
        bundle,                 # <-- FIRST positional arg
        train_split="train",
        val_split="val",
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_size=8,
        max_epochs=5,
        collate_fn=graph_collate,
    )


if __name__ == "__main__":
    main()
