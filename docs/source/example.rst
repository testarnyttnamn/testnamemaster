Example
==================

Here you can find a simple example of how to run the Euclid likelihood as an external likelihood using the Bayesian Analysis tool **cobaya** (check its `docs  <https://cobaya.readthedocs.io/en/latest/index.html>`_). You can run it from an interative python interpreter (**jupyter notebook**) or **shell**.

From a Python interpreter
--------------

The external likelihood needs to be invoked from the text file **cobaya** uses. This file consists of a yaml text, which can be translated into python-dictionaries. For instance:

.. code:: python

   # Info is the 'ini file' dictionary
   info = {
      #Which parameters would you like to sample?
      'params': {
        # Fixed
        'ombh2': 0.022, 'omch2': 0.12, 'H0': 68, 'tau': 0.07,
        'mnu': 0.06, 'nnu': 3.046, 'num_massive_neutrinos': 1,
        'ns': 0.9674,
        #To be passed to euclid which likelihood to use (preliminary)
        # 1: shear
        # 2: spec
        # 12: both
        'like_selection': 12,
        # Sampled - just as an example we assume we will be sampling over A_s
        'As': {'prior': {'min': 1e-9, 'max': 4e-9}, 'latex': 'A_s'}},
      #Which theory code you want to use? HERE CAMB
      'theory': {'camb': {'stop_at_error': True}},
      #Which sample do you want to use? HERE I use MCMC for testing
       'sampler': {'mcmc': None},  
      #Where have you installed codes (i.e: CAMB, polychord, likelihoods...)
      'modules': modules_path,
      #Where are the results going to be stored?
      'output': 'chains/my_euclid_experiment',
      #Likelihood: we load the likelihood as an external function
      'likelihood': {'euclid': "import_module('likelihood.cobaya_interface').loglike"},
    }

The dictionary above has several  *keys*:

- A ``params`` key: parameters that are going to be explored (or derived). They will most of the times computed from the ``theory``code (i.e: **CAMB**). If sampled, you can decide their ``prior``, the the Latex label that will be used in the plots, the reference (``ref``) starting point for the chains (optional), and the initial spread of the MCMC covariance matrix ``proposal``.
- A ``theory`` key: Boltzmann Solver we want to use (i.e: **CAMB** or **CLASS**) to compute theoretical quantities.
- A ``sampler`` key: block stating that we will use the ``mcmc`` sampler to explore the prior+likelihood described above, stating the maximum number of samples used, how many initial samples to ignore, and that we will sequentially refine our initial guess for a covariance matrix. Another samplers such as ``polychord``are accepted.
- A ``modules`` key: path where your external codes (i.e: **CAMB** or **polychord**) are installed. If they are not installed using the structure **cobaya** automatically creates when using automatic installation, you can give particular paths to each of the codes in the corresponding *key*.
- An ``output`` key: path where the products will be written and a prefix for their name.
- A ``likelihood`` key: likelihood pdf's to be used. In this case, we call an external likelihood file that returns the loglike given the ``params`` values. 


To run **cobaya** from the **jupyter notebook** do:

.. code:: python

   from cobaya.run import run
   updated_info, products = run(info)

The **jupyter notebook** ``likelihood_test.ipynb`` is available to play with. In this case, the wrapper *model* from **cobaya** is used. For further information check the `model  <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_  documentation of **cobaya**.

From shell
--------------

The dictionary can be translated into a yaml file:

.. code:: yaml
   
   params:
      ombh2: 0.022
      omch2: 0.12 
      H0: 68
      tau: 0.07
      mnu: 0.06 
      nnu: 3.046 
      num_massive_neutrinos: 1
      ns: 0.9674
      like_selection: 12
      As: 
         prior: 
            min: 1e-9 
            max: 4e-9 
         latex: A_s
   theory:
      camb:
         stop_at_error: True
   sampler:
      mcmc:
   modules: /where/are/modules
   output:  /where/want/output
   likelihood:
      euclid: "import_module('likelihood.cobaya_interface').loglike"


To run **cobaya** simply save the above yaml into a file called ``info.yaml`` and type on shell:

.. code:: bash

   $ cobaya-run info.yaml

   

 
