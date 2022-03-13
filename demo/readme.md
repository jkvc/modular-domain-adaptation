# Demos

This directory contains two ipython notebook demonstrating how to use the main APIs to:
- Train models using the proposed modular domain adaptation techniques
- Evaluate models with in and out of domain data
- Predict with trained models with partially labaled new domain data

## Prerequisites 

To run these notebooks, please follow the setup guide and data acquisition as documented in the [experiments](../experiments/) section.

## Training deep models

Because RoBERTa and other deep model training is prohibitively expensive without GPUs, it is not demonstrated here. The data API for training a general model is the same. Please use `mda.roberta.train_roberta` as an example. 
