# modular-domain-adaptation

This is a framework to train and evaluate models using modular adaptation techniques as proposed in the paper Modular Domain Adaptation (TODO add link).

<img src="paradigm-diagram.png" width="500" align='center'>

Using this framework, the model producer trains models using anticipatory domain adaptation techniques, and the model consumer uses the same techniques when evaluating and predicting with the model on a new domain, to address the domain misalignment issue without the need of releasing the training data. Please see the paper for further details.

## Installation

Clone this repository and use pip at the repo root

```
pip install -e .
```

We recommend using conda. All code is developed using python 3.8. Create a conda environment and install the library within it:

```
conda create -n mda_env python=3.8
conda activate mda_env
pip install -e .
```

Test out installation is successful, run the following to produce no output:

```
python -c 'from mda.api import train_lexicon'
```

CUDA GPUs are supported and used if detected. When using newer CUDA versions, please first install the correct version of `torchvision` with as instructed at pytroch's [installation page](https://pytorch.org/get-started/locally/) using pip, before using `pip install -e .`.

## Getting Started

Please see the [demo section](./demo) for usage of the main APIs.

## Pretrained Models

We [release](https://drive.google.com/drive/folders/1mu2k7PpHoR2Xe_Yyh5CSNfe3WWh0F8ft?usp=sharing) a logistic regression and a RoBERTa model pretrained using all domains in the train split of Media Frames Corpus. 

 - The logistic regression model is released in the form of a lexicon csv, it is trained using domain-specific normalization and domain-specific bias.
 - The RoBERTa model in the form of a torch state dict checkpoint, it is trained using domain-specific bias. 

Please follow the example in `demos` to load, eval, and predict witht them. 

## Citing Modular Domain Adaptation

If you find this work useful, please include a citation to the following publication:

```
@inproceedings{chen.2022.modular,
  author={Junshen K. Chen and Dallas Card and Dan Jurafsky},
  title={Modular Domain Adaptation},
  booktitle={Findings of ACL},
  year={2022},
}
```
