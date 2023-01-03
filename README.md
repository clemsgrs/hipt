<h1 align="center">Hierarchical Image Pyramid Transformer</h2>


Re-implementation of original [HIPT](https://github.com/mahmoodlab/HIPT) code. 

<p>
   <a href="https://github.com/psf/black"><img alt="empty" src=https://img.shields.io/badge/code%20style-black-000000.svg></a>
   <a href="https://github.com/PyCQA/pylint"><img alt="empty" src=https://img.shields.io/github/stars/clemsgrs/hs2p?style=social></a>
</p>

## Requirements

install requirements via `pip3 install -r requirements.txt`

## Prerequisite

You need to have extracted square regions from each WSI you intend to train on.<br>
To do so, you can take a look at [HS2P](https://github.com/clemsgrs/hs2p), which segments tissue and extract relevant patches at a given pixel spacing.


download HIPT pre-trained weights via:

```
mkdir checkpoints
cd checkpoints
gdown 1Qm-_XrTMYhu9Hl-4FClaOMuroyWlOAxw
gdown 1A2eHTT0dedHgdCvy6t3d9HwluF8p5yjz
```

## Step-by-step guide

1. [Optional] Configure wandb

If you want to benefit from wandb logging, you need to follow these simple steps:
 - grab your wandb API key under your profile and export
 - run the following command in your terminal: `export WANDB_API_KEY=<your_personal_key>`
 - change wandb paramters in the config file under `config/` (mainly `project` and `username`)

2. Extract features

Create a configuration file under `config/feature_extraction/` taking inspiration from existing files.<br>
A good starting point is to use the default configuration file `config/default.yaml` where parameters are documented.

Extract region-level features : take a look at `config/feature_extraction/global.yaml`.<br>
Make sure `level` is set to 'global'.<br>

Extract patch-level features : take a look at `config/feature_extraction/local.yaml`.<br>
Make sure `level` is set to 'local'.<br>

Then run the following command to kick off feature extraction:

`python3 extract_features.py --config-name <feature_extraction_config_filename>`

This will produce one .pt file per slide and save it under `output/<dataset_name>/<experiment_name>/<level>/`:

```
hipt/
├── output/<dataset_name>/<experiment_name>/
│     └── level/
│          ├── slide_1.pt
│          ├── slide_2.pt
│          └── ...
```

3. Train **single-fold** model on extracted features

Once features have been extracted, create a configuration file under `config/training/` taking inspiration from existing files.<br>
Then, run the following command to kick off model training on a single fold:

`python3 train.py --config-name <training_single_fold_config_filename>`

4. Train **multi-fold** model on extracted features

Your multiple folds should be structured as follow:

```
fold_dir/
├── fold_1/
│     ├── train.csv
│     ├── tune.csv
│     └── test.csv
├── fold_2/
└── ...
```

Create a configuration file under `config/training/` taking inspiration from `config/multi.yaml`.<br>
Remember to indicate the root directory where your folds are located under `data.fold_dir`.<br>
Then, run the following command to kick off model training on multiple folds:

`python3 train_multi.py --config-name <training_multi_fold_config_filename>`

## Resuming experiment after crash / bug

If, for some reason, feature extraction crashes, you should be able to resume from last processed slide simply by turning the `resume` parameter in your feature extraction config file to `True`, keeping all other parameters unchanged.

## TODO List

- [ ] improve documentation
- [ ] when switching `img_size` argument from `[224]` to `[256]`, make sure the positional enmedding is trainable!
- [ ] switch back optimizer.zero_grad( ) before .step( )
