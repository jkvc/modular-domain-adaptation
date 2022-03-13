# Experiments

This folder contains workflow code for running experiments and producing metrics as present in the paper. This file briefly go over how to run the workflows to reproduce paper results with an example using the Media Frame Corpus (`mfc`).

## Prerequisites

To run experiment workflows, first setup environment and and install dependencies. Clone the entire repository and run the following at the repo root:

```
conda create -n mda python=3.8
conda develop .
pip install -e .
```

## Data

This example uses the Media Frame Corpus. Put the dataset at the correct path. See `repo_root/data_raw` for details on acquiring the data.

Resultant dir structure should be

```
repo_root
    data_raw
        mfc
            climate_labeled.json
            climate_test_sets.json
            deathpenalty_labeled.json
            deathpenalty_test_sets.json
            guncontrol_labeled.json
            guncontrol_test_sets.json
            immigration_labeled.json
            immigration_test_sets.json
            police_labeled.json
            police_test_sets.json
            samesex_labeled.json
            samesex_test_sets.json
            tobacco_labeled.json
            tobacco_test_sets.json
```

Then ingest the data to `mda` defined json format

```
python ../data_ingest/ingest.py ingestor=mfc
```

This produces the formatted dataset json

```
repo_root
    data
        mfc.train.json
        mfc.test.json
```

## Run Experiment Workflows

We use [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) to handle configuration for data, models, etc. And [hydra](https://hydra.cc/docs/intro/) to perform multi-runs with different parameters. See their respective documentation for more details.

All results are saved to `repo_root/wkdir/`.

### All domains

Train a model with all available data in all domains, use the test split to evaluate:

```
python train_all.py -m data_collection=mfc +common=logreg 'model=glob(logreg*)'
python train_all.py -m data_collection=mfc +common=roberta 'model=glob(roberta*)'
```

### Single domain

Train with a single domain's samples, use samples from the remaining domains to evaluate the model

```
python train_single_domain.py -m data_collection=mfc +common=logreg 'model=glob(logreg*)'
python train_single_domain.py -m data_collection=mfc +common=roberta 'model=glob(roberta*)'
```

### Holdout domain

Train with D-1 domains of samples where D is the number of available domains, and use the remaining domain to evaluate the model

```
python train_holdout_domain.py -m data_collection=mfc +common=logreg 'model=glob(logreg*)'
python train_holdout_domain.py -m data_collection=mfc +common=roberta 'model=glob(roberta*)'
```

### Class Distribution Estimation

With the models checkpoints trained in "Holdout domain", run prediction with the model with a class distribution approximated from a small subset of samples from the held out domain, then evaluate on the held out domain.

```
python eval_holdout_domain_class_dist_est.py -m \
    data_collection=mfc \
    model=logreg_dsbias_dsnorm \
    +common=logreg \
    n_labeled_samples=100,150,200,250,300,350,400
python eval_holdout_domain_class_dist_est.py -m \
    data_collection=mfc \
    model=roberta_dsbias \
    +common=roberta \
    n_labeled_samples=100,150,200,250,300,350,400
```

### Roberta Fine-tuning

TBD

## Collect and reduce metrics

All accuracy numbers from the experiment trials above are stored at the leaf level as `**/acc.json`. We recursively mean-reduce accuracy at each directory level to observe the numbers.

```
python acc.py wkdir/
```

This produces `repo_root/wkdir/*_acc.json`. Read or use jq on the json files for numeric results at each reduction level. See `acc.py` for details on the recursive definition of metric and reduction logic.

## Tips

- To use this code base to run experiment with new data, simply define a new ingestor to ingest the new data to the correct `DataCollection` class to be saved as json, then add the necessary configuration files in `config`, and repeat similar run commands.
- If CUDA out of memory, reduce `batch_size` in `config/dataset/roberta.yaml`.
