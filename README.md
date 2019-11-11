# Pytorch code for benchmarking different weakly supervised pixel-wise localization methods with application to histology images for the paper: `Deep weakly-supervised learning methods for classification and localization in histology images: a survey`.

## Arxiv: [https://arxiv.org/abs/1909.03354](https://arxiv.org/abs/1909.03354)
## If you use any part of this code, please cite our work:
```
@article{rony2019weak-loc-histo-survey,
  title={Deep weakly-supervised learning methods for classification and localization in histology images: a survey},
  author={Rony, J. and Belharbi, S. and Dolz, J. and Ben Ayed, I. and McCaffrey, L. and Granger, E.},
  journal={coRR},
  volume={abs/1909.03354},
  year={2019}
}
```

# Code for datasets split/sampling (+ patches sampling from WSI):
* See [./datasets-split](./datasets-split).
* Detailed documentation: [./datasets-split/README.md](./datasets-split/README.md).

# Experiments code:

This repository contains the python codes to run the experiments. In order to ba able to run them, you either need to add the data, fold and results directories inside the directory, modify the location in which the scripts are looking for the data / folds or use the command-line to modify the location of the data / folds and logging directory.

The structure of the data, folds and results directories should be:
```
.
├── data
│   ├── ICIAR2018_BACH_Challenge
│   │   └── Photos
│   │       ├── Benign
│   │       └── ...
│   ├── BreakHis
│   │   └── mkfold
│   │       ├── fold1
│   │       └── ...
│   ├── GlaS
│   │   ├── Grade.csv
│   │   ├── testA_1.bmp
│   │   ├── testA_1_anno.bmp
│   │   ├── testA_2.bmp
│   │   └── ...
│   └── camelyon16
│       ├── w-512xh-512
│       │   ├── metastatic-patches
│       │   └── normal-patches
│       ├── w-768xh-768
│       └── w-1024xh-1024
├── folds
│   ├── bach
│   │   ├── split0
│   │   │   ├── fold0
│   │   │   │   ├── test_s_0_f_0.csv
│   │   │   │   ├── train_s_0_f_0.csv
│   │   │   │   └── valid_s_0_f_0.csv
│   │   │   ├── fold1
│   │   │   └── ...
│   │   └── split1
│   ├── breakhis
│   │   └── ...
│   ├── glas
│   │   └── ...
│   └── camelyon16
│       └── ...
├── results
│   ├── temp    # used to save model before sacred copies them to the experiment directory
│   └── ...
└── ...
```

This repository uses the [sacred package](https://github.com/IDSIA/sacred) to run and watch the experiments. With this package comes a command line interface (see [sacred documentation](https://sacred.readthedocs.io/en/latest/)).
To run an experiment, you can use the command line interface to specify where to save the results, which model to use, etc. For example, to run the localization_mil.py with the CAM - Average model on GlaS:

```python
python localization_mil.py -F results with dataset.glas model.average
```
You can also modify the config of the different _ingredients_ directly from the command line:
```python
python classification_mil.py -F results with dataset.bach model.wildcat model.kmax=1
```
