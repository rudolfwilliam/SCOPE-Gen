# Sequential Conformal Admissibility Control for Generative Models (SCOPE-Gen)

Official code for the ICLR 2025 paper *Conformal Generative Modeling With Improved Sample Efficiency Through Greedy Filters* by Klaus-Rudolf Kladny, Bernhard Schölkopf and Michael Muehlebach.

<p align="center">
<img src="/assets/SCOPE_GEN.svg" width="700">
</p>

## Code Organization

```
.
├── scope_gen
│     ├── algorithms
│     │     └── base.py          # the "heart" of the algorithm: Prediction pipeline generation
│     ├── models                 # prediction pipeline classes
│     ├── calibrate              # calibration functions
│     ├── data                   # basic data structures
│     ├── scripts                # scripts to process data and create tables and figures
│     ├── baselines
│     │     └── clm              # adaptation of the CLM algorithm
│     ├── mimic_cxr
│     │     ├── data             # generative model outputs
│     │     ├── scripts          # evaluation scripts
│     │     ├── configs          # config files for the evaluation scripts
│     │     └── paths.py        
│     ├── ...                    # all other experiments are identical in structure to mimic_cxr
│     ├── nc_scores.py           # non-conformity score functions
│     ├── order_funcs.py         # order functions: determine order according to sub-sample function
│     ├── admissions.py          # admission functions
│     └── distances.py           # distance functions
└── ...
```

## General

### Set up a conda environment

We recommend you to use a conda environment. To install all packages into such an environment, run

```bash
conda env create -f environment.yml -n scope_gen
```
Then, activate the environment
```bash
conda activate scope_gen
```
We note that all required packages will be drawn from the community-led [conda-forge channel](https://conda-forge.org/).
### Run the code

In the current implementation, SCOP-Gen requires three `.npy` files:

- **`scores.npy`**
- **`labels.npy`**
- **`diversity.npy`**

The `scores.npy` array contains the quality estimates for each sample. The `labels.npy` array contains the admissibility labels (`0` means "inadmissible", `1` means "admissible"). The `diversity.npy` array contains similarities (not distances) between samples. The `scores.npy` and `labels.npy` are numpy arrays of shapes `(n, max)`, where n is the amount of calibration points and `max` is the sample limit. `diversity.npy` must be of shape `(n, max, max)`. 

If you want to run the MIMIC-CXR experiment, these files must be moved into

```bash
scope_gen/mimic_cxr/data/outputs
```

After specifying these arrays, you are ready to get started! First, you must format the data. For instance, if you want to reproduce our MIMIC-CXR results, run

```bash
python -m scope_gen.mimic_cxr.scripts.format_data
```

Then, you can reproduce our quantitative evaluation results (including all baselines) via

```bash
python -m scope_gen.mimic_cxr.scripts.eval_all
```

For the qualitative evaluation results, run the jupyter notebook

```bash
jupyter notebook scope_gen/mimic_cxr/scripts/qualitative_comparison.ipynb
```

If you want to reproduce the other experiments, simply replace `mimic_cxr` by any of the other project directories `cnn_dm`, `triviaqa` or `molecules`. The folder structures are identical.

If you would like to reproduce the table in Appendix H, run

```bash
python -m scope_gen.mimic_cxr.scripts.eval --custom_path "single_run_results" --config "./scope_gen/mimic_cxr/scripts/configs/single_runs.json" --name "ourmethod{}" --return_std_coverages True --score "sum"
```

and finally,

```bash
python -m scope_gen.mimic_cxr.scripts.single_runs_assessment
```

## Natural Language Generation Tasks

To generate the numpy files for the tasks `mimic_cxr`, `cnn_dm` and `triviaqa`, follow the instructions of the [CLM auxiliary repository](https://github.com/Varal7/clm_aux).

## Molecular Scaffold Extension

Install the conda environment (assuming you would like to call it `scope_gen_mol`):

```bash
conda env create -n scope_gen_mol -f scope_gen/molecular_extensions/environment.yml
```

### Install DiGress

Clone this specific fork of DiGress ([original repository](https://github.com/cvignac/DiGress)):

```bash
git clone https://github.com/rudolfwilliam/DiGress.git
```

Then, `cd` into the main directory and install it via

```bash
pip install .
```

### Install Moses

Clone the Moses repository

```bash
git clone https://github.com/molecularsets/moses.git
```
and also install this one in the same way as for `DiGress`.

### Obtain the Model Weights

Either obtain model weights by training a DiGress model on the MOSES data train split or simply download the MOSES model checkpoint from the [original repository](https://github.com/cvignac/DiGress). Place the model `.ckpt` file into

```bash
scope_gen/molecules/models/checkpoints
```

### Generate Data

Finally, generate the model predictions via

```bash
python -m scope_gen.molecules.scripts.generate_data
```

Then, you should find the three numpy arrays in the outputs directory.

### Issues?

If you encounter any issues in running the code, please contact me at *kkladny [at] tuebingen [dot] mpg [dot] de*.