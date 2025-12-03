# PyHazard

[![PyPI - Version](https://img.shields.io/pypi/v/PyHazard)](https://pypi.org/project/PyHazard)
[![Build Status](https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazard/docs.yml)](https://github.com/LabRAI/PyHazard/actions)
[![License](https://img.shields.io/github/license/LabRAI/PyHazard.svg)](https://github.com/LabRAI/PyHazard/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/PyHazard)](https://github.com/LabRAI/PyHazard)
[![Issues](https://img.shields.io/github/issues/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)
[![Pull Requests](https://img.shields.io/github/issues-pr/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)
[![Stars](https://img.shields.io/github/stars/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)
[![GitHub forks](https://img.shields.io/github/forks/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)

PyHazard is a Python framework for AI-powered hazard prediction and risk assessment. It provides a modular and extensible architecture for building, training, and deploying machine learning models to predict and analyze natural hazards and environmental risks.

## Features

- **Modular Architecture**: Easy-to-extend framework for implementing custom hazard prediction models
- **Graph Neural Networks**: Built-in support for graph-based data representation and GNN models
- **Flexible Datasets**: Unified dataset interface supporting both DGL and PyTorch Geometric
- **GPU Acceleration**: Full CUDA support for training and inference
- **Extensible Models**: Base classes for implementing custom prediction models
- **Production Ready**: Type hints, proper error handling, and comprehensive testing

## Installation

PyHazard supports both CPU and GPU environments. Make sure you have Python installed (version >= 3.8, <3.13).

### Base Installation

Install the core package:

```bash
pip install PyHazard
```

This will install PyHazard with minimal dependencies.

### CPU Version with Deep Learning Support

```bash
pip install "PyHazard[torch,dgl]" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -f https://data.dgl.ai/wheels/repo.html
```

### GPU Version (CUDA 12.1)

```bash
pip install "PyHazard[torch,dgl]" \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

## Quick Start

Here's a simple example to get started with PyHazard:

```python
import pyhazard
from pyhazard.datasets import Cora
from pyhazard.models.nn import GCN

# Load a dataset
dataset = Cora(api_type='pyg')

# Initialize a model (example using built-in GCN)
model = GCN(
    input_dim=dataset.num_features,
    hidden_dim=64,
    output_dim=dataset.num_classes
)

# Your training and prediction code here
# ...
```

### Using CUDA

To use CUDA for GPU acceleration, set the environment variable:

```shell
export PYHAZARD_DEVICE=cuda:0
```

Or specify the device in your code:

```python
from pyhazard.utils import set_device

set_device("cuda:0")
```

## Documentation

Full documentation is available at: [https://labrai.github.io/PyHazard](https://labrai.github.io/PyHazard)

## Contributing

We welcome contributions! Please see our:
- [Implementation Guideline](.github/IMPLEMENTATION.md) - For implementing new models
- [Contributors Guideline](.github/CONTRIBUTING.md) - For contributing to the project

## Citation

If you use PyHazard in your research, please cite:

```bibtex
@software{pyhazard2025,
  title={PyHazard: A Python Framework for AI-Powered Hazard Prediction},
  author={Cheng, Xueqi},
  year={2025},
  url={https://github.com/LabRAI/PyHazard}
}
```

## License

[BSD 2-Clause License](LICENSE)

## Contact

For questions or contributions, please contact xc25@fsu.edu.
