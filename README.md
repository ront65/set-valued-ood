#  Overview
The overview is removed for the anonymized version. 
# Installation
From this point, we assume you have cloned this repo, and we refer to the root directory of the cloned repo as *root*.

To create a conda environment that works with the repo code, you can run from the root directory:
```bash
source create_wilds_env.sh
```

To run experiments on the Fmow dataset, a separate environment needs to be set (as the Wilds package supports Fmow dataset with different requirements).
```bash
source create_fmow_env.sh
```

This repo supports experiments involving models from the [DomainBed](https://github.com/facebookresearch/DomainBed/tree/dad3ca34803aa6dc62dfebe9ccfb57452f0bb821) codebase.
If you want to run experiments with these models, be sure to install domainbed as a package in your environment.
Otherwise, you should comment out line 25 in utils.py:
```python
#from domainbed.algorithms import IRM, VREx, CORAL, MMD
```

# Download Datasets
To download Wilds datasets, please remove the comment symbol for the relevant dataset in the file *download_datasets.py*, for example:
```python
# Download the Camelyon17 dataset
get_dataset(dataset="camelyon17", download=True)
```
and run the file **from the repo's *root* directory**:
```bash
python download_datasets.py
```
Note: all the data should be installed inside a *data* folder under the repo's *root* directory (This should happen automatically when following the instructions above).

# Usage Examples
To train a model of a specific type (for example, a SET-COVER model) on a specific dataset, one can run an experiment with the necessary flags. \
Then, the results of multiple experiments of a specific dataset can be analyzed and compared using *report.ipynb* notebook.

Details on the supported model types and datasets is given below.

## Running experiments
To run an experiment, one should run the *train.py* file from the repo's *root* directory: `python train.py`.

Experiments can be run with Wandb, in which case results will be written to a specified Wandb project, and can later be used to present the results. 
To run with WANDB, one should add the following flags to the run command (wandb-entity is optional, if not given Wandb will use the default entity set on your server):
```bash
--use-wandb --wandb-project <insert project name> --wandb-entity <insert wandb entity>
```

Below is an example of a full run command:
```bash 
python train.py --use-wandb --wandb-project Camelyon_exp --num-train-domains 20 --num-ood-domains 20 --train-domain-size 6000 --test-domain-size 2000 --seed 10 --model-type set_cover --base-model resnet --dataset camelyon --batch-size 128 --epochs 7 --use-alternative-size-loss
```

### Different experiment types
- To run experiments with SET-COVER, one should add the following flags to the run command: `--model-type set_cover --use-alternative-size-loss `.
- To run experiments with standard baseline (ERM), one should add the following flag to the run command: `--model-type standard`.
- To run experiments with the poolingDCF-CVC split (described in the paper), one should add the following flag to the run command: `--model-type standard --CVC-split`.
- To run experiments with one of the DomainBed models, one should add any of the following flags (These are supported only with *Camelyon*, *Fmow*, and *iWildCam* datasets):
  - For IRM: `--model-type domain_bed --base_model irm`
  - For VREx: `--model-type domain_bed --base_model vrex`
  - For MMD: `--model-type domain_bed --base_model mmd`
  - For CORAL: `--model-type domain_bed --base_model coral`

### Different datasets 
All the examples below run the SET-COVER algorithm.
- Full example command for running an experiment on *Camelyon* dataset:
  ```bash 
  python train.py --use-wandb --wandb-project Camelyon_exp --num-train-domains 20 --num-ood-domains 20 --train-domain-size 6000 --test-domain-size 2000 --seed 10 --model-type set_cover --base-model resnet --dataset camelyon --batch-size 128 --epochs 7 --use-alternative-size-loss
  ```
  Camelyon experiments should be using either resnet model (`--base-model resnet`) or one of the DomainBed models (e.g `--model-type domain_bed --base_model irm`).
  
- Full example command for running an experiment on *Fmow* dataset:
  ```bash
  python train.py -use-wandb --wandb-project Fmow_exp --num-train-domains 20 --num-ood-domains 18 --train-domain-size 4500 --test-domain-size 3000 --seed 10 --model-type set_cover --base-model resnet --dataset fmow --batch-size 64 --epochs 7 --use-alternative-size-loss
  ```
  Fmow experiments should be using either resnet model (`--base-model resnet`) or one of the DomainBed models (e.g `--model-type domain_bed --base_model vrex`).
  
- Full example command for running an experiment on *iWildcam* dataset:
  ```bash
  python train.py --use-wandb --wandb-project Iwildcam_exp --num-train-domains 80 --num-ood-domains 40 --train-domain-size 1000 --test-domain-size 3000 --seed 10 --model-type set_cover --base-model resnet --dataset iwildcam --batch-size 64 --epochs 7 --use-alternative-size-loss
  ```
  iWildCam experiments should be using either resnet model (`--base-model resnet`) or one of the DomainBed models (e.g `--model-type domain_bed --base_model coral`).
  
- Full example command for running an experiment on *Amazon* dataset:
  ```bash
  python train.py --use-wandb --wandb-project Amazon_exp --num-train-domains 500 --num-ood-domains 100 --train-domain-size 1000 --test-domain-size 1000 --seed 10 --model-type set_cover --base-model mlp --dataset amazon --batch-size 128 --epochs 20 --use-alternative-size-loss
  ```
  Amazon experiments should be using mlp model (`--base-model mlp`).
  
- Full example command for running an experiment on *Synthetic* dataset with dimension *d=10*:
  ```bash
  python train.py --use-wandb --wandb-project Synthetic_10_exp --num-train-domains 25 --num-ood-domains 25 --train-domain-size 2000 --test-domain-size 1000 --seed 10 --model-type set_cover --base-model mlp --dataset synthetic_50 --batch-size 128 --epochs 2 --use-alternative-size-loss
  ```
- Full example command for running an experiment on *Synthetic* dataset with dimension *d=50*:
  ```bash
  python train.py --use-wandb --wandb-project Synthetic_50_exp --num-train-domains 25 --num-ood-domains 25 --train-domain-size 2000 --test-domain-size 1000 --seed 10 --model-type set_cover --base-model mlp --dataset synthetic_50 --batch-size 128 --epochs 2 --use-alternative-size-loss
  ```
  Synthetic experiments should be using mlp model (`--base-model mlp`).

## Analysing the results
*report.ipynb* is a jupyter notebook template for analyzing the results of multiple experiments (a single notebook should be used for experiments of a single dataset).
To use the notebook templates, the experiments should be run with Wandb.
