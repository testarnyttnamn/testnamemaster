[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# Likelihood-implementation

## Set-Up

### Stand alone build

The package can be installed by simply running:

```bash
python setup.py install
```

### Conda environment

To build the package in a dedicated Conda environment with development tools run:

```bash
conda env create -f environment.yml
conda activate cosmobox
python setup.py install
```

### Unit tests

To run the unit tests locally run:

```bash
python setup.py test
```

Note that this requires the development tools.
