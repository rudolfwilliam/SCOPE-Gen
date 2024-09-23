# Sequential Conformal Admissibility Control for Generative Models (SCAC-Gen)

Repository for the manuscript *Conformal Generative Modeling With Improved Sample Efficiency Through Greedy Filters*.

# Code Organization

```
.
├── pcgen
│     ├── algorithms
│     │      
│     │
│     ├── celeba
│     │      ├── data
│     │      ├── scm                         # Model classes and training scripts (vae, flow)
│     │      │    ├── config                 # Config files for models
│     │      │    ├── scripts
│     │      │    │     ├── train_flows.py   # Train flow models
│     │      │    │     ├── train_vae.py     # Train vae model
│     │      │    │     └── ... 
│     │      │    └── ...
│     │      ├── baselines                   # Baseline models
│     │      ├── visualizations              # Scripts that reproduce figures from the paper
│     │      └── eval                        # Scripts that evaluate different methods
│     ├── morphomnist
│     └── ...
└── ...
```

# Natural Language Question Answering

For data generation, follow the instructions in

https://github.com/Varal7/clm_aux.git

# Molecular Scaffold Extension

Install the conda environment (assuming you would like to call it `scorgen_mol`):

```console
conda env create -n scorgen_mol -f scorgen/molecular_extensions/environment.yml
```

## Install DiGress

Clone this specific fork of DiGress (original repository: ):

```console
git clone https://github.com/rudolfwilliam/DiGress.git
```

Then, `cd` into the main directory and install it via

```console
pip install .
```

## Install Moses

Clone the Moses repo

```console
git clone https://github.com/molecularsets/moses.git
```
and also install this one in the same way as for `DiGress`.
