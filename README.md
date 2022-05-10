# minian-validation

repo containing all validations and benchmarking of minian

## steps to reproduce

The scripts in this repo try to use a local SSD as intermediate folder when running both pipelines to ensure a fair comparison.
It will create a `~/var/minian-validation` directory under your home folder.
Make sure your home folder have enough space (500G+) and ideally on a SSD.
Otherwise you can change the `*_INT_PATH` variables in `run_*.py` scripts to point it to a better location for storing intermediate variables.

### create environments

There are 3 environments needed to reproduce this repo and you need to use the correct environment for each script.
You can create them like below:

1. `conda env create -n minian-validation-generic -f environments/generic.yml`
2. `conda env create -n minian-validation-minian -f environments/minian.yml`
3. `conda env create -n minian-validation-caiman -f environments/caiman.yml`

### reproduce validations (Figure 15, 16, 17)

#### simulate data

1. `conda activate minian-validation-generic`
2. `python simulate_data_validation.py`

#### run both pipelines on simulated data

1. change the `DPATH` variable in both [run_minian_simulated.py](https://github.com/denisecailab/minian-validation/blob/d594fa68876cbc5a6a59c3e3acecfa7daba1972a/run_minian_simulated.py#L21) and [run_caiman_simulated.py](https://github.com/denisecailab/minian-validation/blob/d594fa68876cbc5a6a59c3e3acecfa7daba1972a/run_caiman_simulated.py#L16) to `"./data/simulated/validation"`
2. `conda activate minian-validation-minian; python run_minian_simulated.py`
3. `conda activate minian-validation-caiman; python run_caiman_simulated.py`

#### run both pipelines on real data

You will need source data that is stored on [figshare](https://doi.org/10.6084/m9.figshare.c.5987038.v1).
We have a convenient script that will help you download all the relevant data and store them in the correct place.

1. `conda activate minian-validation-generic; python get_data.py`
2. `conda activate minian-validation-minian; python run_minian_real.py`
3. `conda activate minian-validation-caiman; python run_caiman_real.py`

#### plot validation results and generate summary output

1. `conda activate minian-validation-generic; python plot_validation.py`

### reproduce benchmarks (Figure 18)

#### simulate data for benchmarking (Optional)

This section will simulate datasets used for benchmarking.
Since the benchmark results are csv files already stored on this repo, you can skip this and the next section if you just want to reproduce the plotting and use the benchmark results we have.

1. `conda activate minian-validation-generic; python simulate_data_benchmark.py`

#### run both pipelines on simulated data (Optional)

1. change the `DPATH` variable in both [run_minian_simulated.py](https://github.com/denisecailab/minian-validation/blob/d594fa68876cbc5a6a59c3e3acecfa7daba1972a/run_minian_simulated.py#L21) and [run_caiman_simulated.py](https://github.com/denisecailab/minian-validation/blob/d594fa68876cbc5a6a59c3e3acecfa7daba1972a/run_caiman_simulated.py#L16) to `"./data/simulated/benchmark"`
2. `conda activate minian-validation-minian; python run_minian_simulated.py`
3. `conda activate minian-validation-caiman; python run_caiman_simulated.py`

#### plot benchmark results and generate summary output

1. `conda activate minian-validation-generic; python plot_benchmark.py`

### reproduce performance tradeoff (Figure 19)

#### simulate data for tradeoff

1. `conda activate minian-validation-generic; python simulate_data_benchmark.py`

#### run both pipelines with varying number of parallel processes

1. `conda activate minian-validation-minian; python run_minian_tradeoff.py`
2. `conda activate minian-validation-caiman; python run_caiman_tradeoff.py`

#### plot tradeoff results and generate summary output

1. `conda activate minian-validation-generic; python plot_tradeoff.py`