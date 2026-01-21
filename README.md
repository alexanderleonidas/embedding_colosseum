# Embedding Strategies for High-Dimensional Imaging in Quantum Machine Learning Pipelines

## Getting Started
To start, download [uv](https://docs.astral.sh/uv/getting-started/installation/) and execute `uv sync` in your terminal. This automatically installs the correct python version and packages. For this to be working packages have to be removed/ added to the requirement files using the following commands:
- To add packages to the project: `uv add PACKAGE_NAME`
- To remove packages to the project: `uv remove PACKAGE_NAME`

To then run a file execute `uv run file_name.py`. This will automatically update the virtual environment and remove or install any packages if changes are detected.

## Defining and running experiments
We use [Hydra.cc](https://hydra.cc/) as a framework to define our experiments and the variables within our pipeline. When calling the `main.py` file hydra is automatically intialized and takes the default configuration as defined in `conf/config.yaml`. Experiments can be defined using the sweep feature and run by appending the configuration to the default one e.g. `uv run main.py +experiment=file_name`. Where experiment is the directory in which the file is located.

## Repository Structure

```
embedding_colosseum/
├── conf/                    # Hydra configurations
├── src/
│   ├── classify_dataset/    # Feature extraction utilities
│   ├── dataset/             # Data loading and management
│   ├── embeddings/          # Quantum embedding implementations
│   ├── model/               # Variational quantum classifier
│   ├── preprocessing/       # Dataset preprocessing
│   ├── training/            # Training pipeline
│   └── utils/               # Logging and metric saving
├── results/                 # Results of final experiments
├── main.py                  # Main entry point
├── ...
└── pyproject.toml           # Project dependencies
```

## Supported Datasets

Downloaded automatically:
- MNIST, Fashion-MNIST, CIFAR-10, STL-10

Theses datasets need to be downloaded separately:
- EuroSAT (RGB and Multispectral)
- Brain Tumor
- CXR8 (Chest X-Ray)

## Embedding Strategies
We plan to use the following embedding strategies in our project:

Quantum Image Representation:
- FRQI: [Flexible Representation of Quantum Images](https://doi.org/10.1007/s11128-010-0177-y) (2011)
- NEQR: [Novel Enhanced Quantum Representation](https://doi.org/10.1007/s11128-013-0567-z) (2013)
- OQIM: [Order-encoded Quantum Image Model](https://doi.org/10.1007/s11128-019-2463-7) (2019)
- NAQSS: [Normal Arbitrary Quantum Superposition State](https://doi.org/10.1007/s11128-013-0705-7) (2014)

Parameterized Quantum Circuits:
- Angle Encoding
- ZZFeatureMap
- RMP
