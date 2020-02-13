[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# Mock structure for COBAYA as external likelihood

### What's COBAYA?

Cobaya (code for bayesian analysis, and Spanish for Guinea Pig) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers (including the advanced MCMC sampler from CosmoMC, and the advanced nested sampler PolyChord).

### COBAYA's documentation

Check [documentation](https://cobaya.readthedocs.io/en/latest/index.html)

### Install COBAYA

Install Cobaya by simply running:

```bash
pip install cobaya --upgrade
```
### Install modules for COBAYA (such as CAMB, CLASS, Polychord...)

Check [instructions](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html)

```bash
cobaya-install cosmo -m /path/to/modules
```

### Calling external likelihood
Open and play with ```notebooks/likelihood_test.ipynb```

### Run unit test
To run the unit tests locally run:

```bash
python setup.py test
```

Note that this requires the development tools.
