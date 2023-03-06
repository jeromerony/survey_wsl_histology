### Pytorch 1.10.0 code for:

`Deep Weakly-Supervised Learning Methods for Classification and Localization in Histology Images: A Survey`. [Webpage](https://www.melba-journal.org/papers/2023:004.html)

### Citation:
```
@article{rony2023deep,
    title ={Deep Weakly-Supervised Learning Methods for Classification and Localization in Histology Images: A Survey},
    author={Rony, J. and Belharbi, S. and Dolz, J. and Ben Ayed, I. and McCaffrey, L. and Granger, E.},
    journal = "Machine Learning for Biomedical Imaging",
    volume = "2",
    year = "2023",
    pages = "96--150"
}
```

### Issues:
Please create a github issue.


### Content:
* [View](#view)
* [Results](#results)
* [Requirements](#re2q)
* [Datasets](#datasets)
* [Run code](#run)



#### <a name='view'> Method</a>:
<img src="doc/view.png" alt="view" width="600">

<img src="doc/taxo.png" alt="taxonomy" width="600">

Installed methods:
- GAP
- Max-Pooling
- LSE
- CAM
- HaS
- WILDCAT
- ACoL
- SPG
- Deep MIL
- PRM
- ADL
- CutMix
- TS-CAM
- MAXMIN
- NEGEV
- GradCAM
- GradCAM++
- Smooth-GradCAM++
- LayerCAM
- U-Net

#### <a name='results'> Results</a>:

<img src="doc/per-loc.png" alt="localization performance" width="600">

<img src="doc/sensitivity-thresh.png" alt="localization performance and its sensitivity to thresholding" width="600">

<img src="doc/impact-model-selecion.png" alt="Impact of model selection over localization performance" width="600">

<img src="doc/glas-benign.png" alt="localization GLAS dataset, Benign." width="600">

<img src="doc/glas-malignant.png" alt="localization GLAS dataset, Malignant." width="600">

<img src="doc/camelyon-normal.png" alt="localization Camelyon16 dataset, Normal." width="600">

<img src="doc/camelyon-metastatic.png" alt="localization Camelyon16 dataset, Metastatic." width="600">


#### <a name='reqs'> Requirements</a>:

* Python 3.7.10
* [Pytorch](https://github.com/pytorch/pytorch)  1.10.0
* [torchvision](https://github.com/pytorch/vision) 0.11.1
```shell
pip install torch==1.10.0 -f https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
```
* [Full dependencies](dependencies/requirements.txt)
* Build and install CRF:
    * Install [Swig](http://www.swig.org/index.php)
    * CRF (not used in this work, but it is part of the code.)

```shell
cdir=$(pwd)
cd dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
cd $cdir
cd dlib/crf/crfwrapper/colorbilateralfilter
swig -python -c++ colorbilateralfilter.i
python setup.py install
```

#### <a name="datasets"> Download datasets </a>:
#### 2.1. Links to dataset:
* [GlaS](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest)
* [Camelyon16](https://github.com/jeromerony/survey_wsl_histology)


#### 2.2. Download datasets:

* GlaS: [./download-glas-dataset.sh](./download-glas-dataset.sh).

You find the splits in [./folds](./folds).


### 2.3 Code for datasets split/sampling (+ patches sampling from WSI):
* See [datasets-split](https://github.com/jeromerony/survey_wsl_histology/tree/init-branch/datasets-split).
* Detailed documentation: [datasets-split/README.md](https://github.com/jeromerony/survey_wsl_histology/blob/init-branch/datasets-split/README.md).

#### <a name="datasets"> Run code </a>:

* CAM-method: CAM over GLAS using ResNet50:
```shell
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid
getfreeport() {
freeport=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
}
export OMP_NUM_THREADS=50
export NCCL_BLOCKING_WAIT=1
plaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
getfreeport
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1  \
                         --master_port=$freeport main_wsol.py \ --local_world_size=1 \
                         --task STD_CL \
                         --encoder_name resnet50 \
                         --arch STDClassifier \
                         --runmode final-mode \
                         --opt__name_optimizer sgd \
                         --batch_size 32 \
                         --eval_checkpoint_type best_localization \
                         --opt__step_size 250 \
                         --opt__gamma 0.1 \
                         --max_epochs 1000 \
                         --freeze_cl False \
                         --support_background True \
                         --method CAM \
                         --spatial_pooling WGAP \
                         --dataset GLAS \
                         --fold 0 \
                         --cudaid 0 \
                         --debug_subfolder None \
                         --amp True \
                         --opt__lr 0.003 \
                         --exp_id 11_19_2021_09_32_36_109051__423849
```
