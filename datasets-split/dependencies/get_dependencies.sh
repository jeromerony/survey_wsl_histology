#!/usr/bin/env bash

#  =============================================================================================
#                                    ANACONDA ENVIRONMENT
#                                 (DO NOT USE THIS, it is old)
# ==============================================================================================


## General:
## see: https://github.com/conda/conda/issues/4339
## Add channel: conda config --add channels conda-forge
#
#source activate pytorch.1.0.0.0
## conda env export | cut -f 1 -d '=' | grep -v "prefix" > pytorch.1.0.0.yml
#conda env export | cut -f 1 -d '='  | grep -v "prefix" > pytorch.1.0.0.yml
#
## Steps:
## 1. Change the name of your virtual environment if you like, the first line in the file pytorch.1.0.0.yml
## 2. Use specific version of the following packages:
##   2.1 python=3.7.1
##   2.2 pytorch=1.0.0
#
## Waning:
## 1. Use source-forge channel.
## 1. Use: blas=*=openblas
## 3. Create an env: conda env create -f pytorch.1.0.0.yml



#  =============================================================================================
#                                    PURE PYTHON 3.7.0 VIRTUAL ENVIRONMENT
# ==============================================================================================

source ~/Venvs/python37OS-openslide/bin/activate
pip freeze > requirements.txt