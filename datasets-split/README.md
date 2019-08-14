# Code for creating datasets split/Sampling:

## Notes:

## Bug report:
Contact: [https://sbelharbi.github.io/contact/](https://sbelharbi.github.io/contact/).

## Note on the datasets:
Splitting, k-folding, sampling from any datset assumes that the original structure of the folders/files has not been changed.

## Concerned datasets:
* [BreaKHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
* [Dataset of BACH Challenge (Part A) 2018](https://iciar2018-challenge.grand-challenge.org/)
* [GlaS Dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest)
* [CAMELLYON16 Dataset](https://camelyon16.grand-challenge.org/Home/)

Our splits/folds can be found in `./folds`. The `CAMELYON16` splits/folds can be found in `./folds/camelyon16/WSI-level/` (for the WSI level, and in `./folds/camelyon16/w-?-x-h-?/patches-level-fourth-pass` (for the patch level. Replace `?` with the corresponding patch size.)

## Reproducibility:
The code is deterministic. Running the code twice will provide exactly the same result. See the module: `./reproducibility.py` to see the concerned modules with determinism.

The determinism in the code is controlled using a provided seed. The default seed is `0`. You can use your own seed by
creating an environment variable in `MYSEED` in `Bash` and specifiying its value (in this example, the value is `0`):
```sh
$ export MYSEED=0
```
Then, run the code. We used in our experiments the default seed (i.e., `MYSEED=0`) which is already setup in the code.

###  ===== WARNING ON REPRODUCIBILITY ===== :

If you care about reproducibility:
  * Case of sampling patches from `camelyon16` with size `(w_, h_)`:
  If you run the script (`wsi.py`) for one single size `(w_, h_)`, then the
  script stopped for some reason, the script will continue without processing again the processed samples.  Despite this, the script will still produce the same results as if the script runs without any interruption. The other thing we want to mention is that the script is re-seeded to the default seed for each size `(w_, h_)`, this will allow to run the script with many different sizes at once.
  * Using more than one process (multiprocessing case): In this case the reproducibility is out of the question unless the same number of processes is used (we provide below a solution to deal with this issue). This means that the two following runs will lead to two different results: `$ python wsi.py 10` and `$ python wsi.py 20`. The reason of the non-reproducibility in this case is that each process will inherent the same state of the random generator of the parent process. Since the WSIs are processed in batch where each process deals with a batch, an image `i` in batch `j` will be processed by the process with a state of the random generator `(i, j)`. As you can see, the state depends on the batch (process) and the rank of the image in the batch. Our solution to this problem is to define a seed `s` for each image 'i' beforehand (i.e., within the parent process. The seed is sampled randomly using `numpy.random.randint(0, MAX_SEED + 1)`). Then, re-seed each process before it process each image to the seed of the image in hand. The bottom line, the code is reproducible independently of the number of processes.
  * We sample the same number of normal patches as the metastatic ones, when possible. For a set within a split/fold, we sample first the necessary metastatic patches (since there is only few of them). Let's say we obtained `n` metastatic patches for this set. Now, we need to sample `n` normal patches from normal WSI (assuming there is `k` normal WSI in this set). Since there is no prior to tell us to sample from which WSI, we decide to sample `n/k` normal patch per WSI. Now, in most the cases this is poissible. But, some WSI contain very small tissue region, and they can not provide such number of patches (particularly when sampling without overlapping). What we did is to sample the possible patches, assuming `s` form the WSI (which will be less than `n/k`), then ask the process to sample `s + n/k` patches from the next WSI in the batch. This goes under the risque that there may be a left over at the end of processing a batch (which is the case in out experiment). Using `48` process will lead to a very few missing normal patches (1 or 2 patches). This is a very minor issue. The bottom line, if you want to reproduce EXACTLY our sampling results, you need to use `48` processes when running `wsi.py`.


## Sampling from `CAMELYON16` dataset:

### Example of patches with different sizes:
* `(w, h) =  (512, 512)`:
  * Normal (WSI: `training/normal/normal_001.tif`):
  <center>
  <img src="./plot-samples/512x512/normal.jpeg" alt="Normal patches (WSI: `training/normal/normal_001.tif`) with size `(w, h) =  (512, 512)`" title="Normal (WSI: `training/normal/normal_001.tif`) with size `(w, h) =  (512, 512)`"
  width="400"/>
  </center>

  * With turmor (WSI: `training/tumor/tumor_001.tif`) [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)]:
  <center>
  <img src="./plot-samples/512x512/tumor.jpeg" alt="Patches with tumor [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)] (WSI: `training/tumor/tumor_001.tif`) with size `(w, h) =  (512, 512)`" title="Patches with tumor [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)] (WSI: `training/tumor/tumor_001.tif`) with size `(w, h) =  (512, 512)`"  width="400"/>
  </center>

* `(w, h) =  (768, 768)`:
  * Normal (WSI: `training/normal/normal_001.tif`):

  <center>
  <img src="./plot-samples/768x768/normal.jpeg" alt="Normal patches (WSI: `training/normal/normal_001.tif`) with size `(w, h) =  (768, 768)`" title="Normal (WSI: `training/normal/normal_001.tif`) with size `(w, h) =  (768, 768)`"
  width="400"/>
  </center>

  * With turmor (WSI: `training/tumor/tumor_001.tif`) [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)]:

  <center>
  <img src="./plot-samples/768x768/tumor.jpeg" alt="Patches with tumor [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)] (WSI: `training/tumor/tumor_001.tif`) with size `(w, h) =  (768, 768)`" title="Patches with tumor [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)] (WSI: `training/tumor/tumor_001.tif`) with size `(w, h) =  (768, 768)`"  width="400"/>
  </center>

* `(w, h) =  (1024, 1024)`:
  * Normal (WSI: `training/normal/normal_001.tif`):

  <center>
  <img src="./plot-samples/1024x1024/normal.jpeg" alt="Normal patches (WSI: `training/normal/normal_001.tif`) with size `(w, h) =  (1024, 1024)`" title="Normal (WSI: `training/normal/normal_001.tif`) with size `(w, h) =  (1024, 1024)`"
  width="400"/>

  </center>

  * With turmor (WSI: `training/tumor/tumor_001.tif`) [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)]:

  <center>
  <img src="./plot-samples/1024x1024/tumor.jpeg" alt="Patches with tumor [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)] (WSI: `training/tumor/tumor_001.tif`) with size `(w, h) =  (1024, 1024)`" title="Patches with tumor [Top row: patches. Bottom row: masks (white area indicates metastatic pixels)] (WSI: `training/tumor/tumor_001.tif`) with size `(w, h) =  (1024, 1024)`"
  width="400"/>
  </center>


### Memory usage and speed:
The code supports multiprocessing using the Python native module [multiprocessing](https://docs.python.org/3.7/library/multiprocessing.html). One process can work with `1024MB` of memory. If you use many processes, consider using large memory.

### Requirements:
General requirements:
* [Python 3.7.1](https://www.python.org/downloads/release/python-371/)
* [Openslide 3.4.1](https://openslide.org/)
* [Openslide-Python 1.1.1](https://openslide.org/api/python/)
* [Pytorch nightly build dev: 1.2.0.dev20190616](https://pytorch.org/) (only fro some post-processing-visualization stuff, nothing to do with sampling)
* [Torchvision]() (ditto)

To visualize WSI, here is a useful tool: [ASAP (Automated Slide Analysis Platform)](https://github.com/computationalpathologygroup/ASAP) [Download](https://github.com/computationalpathologygroup/ASAP/releases). it requires Openslide.

All dependencies can be found at [./dependencies/requirements.txt](./dependencies/requirements.txt). To install:
```sh
$ python
Python 3.7.1 (default, Oct 22 2018, 11:21:55)
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
$ python -m venv /path/to/new/virtual/environment
$ source /path/to/new/virtual/environment/bin/activate
$ pip install -r ./dependencies/requirements.txt
$ # Install Pytorch
$ pip install https://download.pytorch.org/whl/nightly/cu100/torch_nightly-1.2.0.dev20190616-cp37-cp37m-linux_x86_64.whl
$ # Install torchvision
$ pip install torchvision==0.2.2.post3
```

### How To sample (General process)?
1. Metastatic patches: It is done in two passes: (This step is done independently of the splits/folds (it is done only once).)
    1. Passe 1: Metastatic patches are sampled: all patches that have enough tissue, and percentage of cancerous  pixel  `>= p0` are selected. At this moment, the patches are stored on disc.
    2. Pass 2 (calibration): We select a subset of the metastatic patches while considering the following: The number of the metastatic patches is calibrated in a way to obtain equally distributed  patches with few/large percentage of cancerous pixels. Since, the number of patches with `100%` of metastatic pixels will be large compared with the number of patches with few metastatic pixels, a calibration is required otherwise the localization of the metastatic regions is meaningless (since most the patches have `100%` of metastatic pixels. Therefore, whatever the model's prediction it will hit correct pixels --> meaningless localization).

2. Now, we sample normal patches: (normal patches are sampled only from normal WSI.)
    * For each split:
        * For each fold:
            * For each set in [train, valid, test]:
                * Let's consider `n` is the number of normal WSIs within this set.
                * Count the number of cancerous patches (let it be `m`).
                * Sample `n/m` normal patch per WSI (normal).
3. How to check the tissue mask within a patch? we use a binarization approach [(OTSU method)](https://en.wikipedia.org/wiki/Otsu's_method), then, compute the tissue mass. If the tissue mass is above a certain threshold, the patch is accepted. In all the experiments, we used the threshold `10%` of the entire mass of the patch. We used the function [skimage.filters.threshold_otsu()](http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu) in practice.

### General rules to accept a patch:
* Metastatic patch is accepted if:
  * It has enough tissue. (at `level=0`)
  * It has enough metastatic pixels (at `level=0`).
* Normal patch is accepted if:
  * It has enough tissue (at `level=6`. Why? at `level=0` it seems that the binarization of a patch is not enough to
  detect tissue part from background (white matter) while it is enough at `level=6` (see image below of the binarization of `training/normal/normal_001.tif` at `level=6`)).
  Therefore, the search of an acceptable patch is done at `level=0`. If it is accepted, we double check at `level=0`. If it is accepted, we take the patch at `level=0`.

  <center>
  <img src="./doc-img/normal-bin-level-6/img_rgb.png" alt="training/normal/normal_001.tif at level 6"
  title="training/normal/normal_001.tif at level 6" width="200"/>
  <img src="./doc-img/normal-bin-level-6/tissue_mask.png" alt="OTSU binary mask of training/normal/normal_001.tif at
  level 6" title="OTSU binary mask of training/normal/normal_001.tif at level 6" width="200"/>
  </center>


### Metastatic patches calibration:
The percentage of metastatic patches varies from patch to another depending on its position with respect to the tumor. Patches on the edge of the tumor have less metastatic pixels, while patches within the tumor are composed almost
entirely from metastatic pixels. Sampling patches from a cancerous region will often lead to a large number of patches with high percentage of metastatic pixels since the surface of the middle of the tumor is way bigger than the
 surface of its edge. Since we intend to use these patches for cancerous regions localization, patches with high
 percentage of metastatic pixels may shadow the model's capacity to localize the metastatic regions, and make the  evaluation of localization useless. To avoid this, we calibrate the number of patches. We create two categories:

* `Category A`: consists of all patches with `p0 <= p <= p1`, and contains `N_A` patches.
* `Category B`: consists of all patches with `p > p1`, and contains `N_B` patches.

  Only patches in `Category B` are calibrated. Patch in `Category A` are taken as they are. We pre-define a percentage
   `n`. The number of patches that we take from `Category B` is equal to: `N_A * n`. Therefore, we sample uniformly (without repetition) `N_A * n` patches from `Category B` from the bins of the histogram of the frequency of the percentage of the metastatic pixels in all  metastatic patches. The non-selected of the patches of `Category B` is thrown away.

#### Note on the calibration:
Since the original train and test sets (WSIs) are likely to have different distribution of the two categories, we calibrate each set separately.

Example on the train set (`n = 0.01 = 1%`, patch size: `(w_, h_)=(512, 512)`):
* **Before calibration**:
<center>
<img src="./folds/camelyon16/w-512xh-512/patches-level-first-pass/training/BEFORE-CALIBRATION-train-set-patch-w-512-h512.png" alt="train set before calibration" title="train set before calibration" width="400"/>
</center>

* **After calibration**:
    <center>
    <img src="./folds/camelyon16/w-512xh-512/patches-level-first-pass/training/AFTER-CALIBRATION-train-set-patch-w-512-h512.png" alt="train set After calibration" title="train set After calibration" width="400"/>
    </center>

We note that patches with `0.2 < p < 0.3` are more frequent. This is due to the fact that such patches are the ones located on the edge. Therefore, they have more chance to be sampled since we ALWAYS start sampling metastatic patches
at the edge. The way we approach the edge of a tumor is using a window stride of `1` (very SLOW window sliding. This increases the chance to obtain patches with a percentage of metastatic patches just a little bit above the lower bound
(i.e., `0.2`). As a consequence of this slow approach to the edge to find the first acceptable metastatic patch, the following sampled patches within the same row are more likely to have a very high percentage of metastatic patches
(around `1` since the next sampled patches are in the center of the tumor) due to the non-overlap slided window. This explains why patches `0.5 <= p <= 0.9` are rare.

In all the experiments, we used `n = 1%`.



### Sampling normal patches:
1. We do not sample all possible patches from the WSI, since it is unnecessary, and it is time consuming. Instead, for each split, for each fold, for each set (train/valid), for each WSI, we sample the required number of patches. The sampling is done randomly without repetition.
2. Since not all the patches within a WSI are acceptable (most of the WSI is white: the glass material), we select our patches.
3. We target in our sampling the tissues. The tissue is clearly visible through a binarization at `level=6`. Therefore, we perform the random search of acceptable patches in `level=6`. The patch needs to have enough tissue. Once we find a patch, we map its coordinates into `level=0` and grab it. We double-check if the patch still acceptable at this level. If the patch is acceptable, we take it.



### Notes on sampling from `CAMELYON16` dataset:
1. Sample `test_114` was discarded entirely. It is annotated as cancerous, but there is no `*.xml` annotation file.
2. The splits/k-fold are first performed on WSI-files level to avoid that patches from one WSI end up in train, valid, and test sets.
3. How to deal with nested metastatic regions? Let us consider `k` metastatic regions that are nested within a large
region `R`. In this case, we process only the largest region (i.e., the region `R`). [in order to speedup computation].
However, when  drawing the metastatic mask of the largest region, the nested regions are considered.
4. We use `PIL.ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)` [(PIL)](https://pillow.readthedocs.io/en/5.1.x/reference/ImageDraw.html#PIL.ImageDraw.PIL.ImageDraw.Draw) to draw the metastatic mask. It is WAY faster than `skimage.measure.points_in_poly()` [(Scikit-image)](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.points_in_poly).
5. We use the Python module `multiprocessing` for multi-processing the sampling from the WSIs. Each process deals
with a batch of WSIs.
6. We focus on sampling patches around the edges of the tumor in order to obtain large number of patches with few metastatic pixels (there are a large number of patches sampled with `100%` of metastatic patches. They are usually sampled from the center of the tumor region).
7. Sampling metastatic patches are extremely light in term of memory usage (few Megabytes per WSI. Although, the required memory depends on the WIDTH of the tumors since we split the tumor region into rows and load each row
sequentially. We run all the script with different patch size using only `1024MB` of memory per process.), and reasonably fast [Mainly, the speed depends on the number and size of the non-nested tumors].
8. Sampling is performed on the highest resolution of the WSI (i.e., `level=0`).
9. If a WSI has already been processed (we check the `*.csv` files), we do not process it again. This means, if the script stopped for any reason, it does not start over, but it continues.
10. While the code allows sampling metastatic patches with overlap, in our experiments, the patches are sampled without overlap: `delta_w = w_` and `delta_h = h_`.
11. Sampling metastatic patches from WSIs:
    * Sampling ALL the metastatic patches with a specific size takes less than `03h30min` using `48` processes with `1024MB` of memory per each process.
    * #### Size `(w_=512, h_=512)`:
        * **Total patches**:
          * **BEFORE CALIBRATION**: 137,769 where 14,912 (10.82%) are in 0.2 <= p <= 0.5, and 122,857 (89.18%) are in p > 0.5.
          * **AFTER CALIBRATION**: 24,435 where 14,912 (61.03%) are in 0.2 <= p <= 0.5, and 9,523 (38.97%) are in p > 0.5.
        * **Original train WSI**:
          * **BEFORE CALIBRATION**: 76,327 where 10,089 (13.22%) are in 0.2 <= p <= 0.5, and 66,238 (86.78%) are in p > 0.5.

          <center>
          <img src="./folds/camelyon16/w-512xh-512/patches-level-first-pass/training/BEFORE-CALIBRATION-train-set-patch-w-512-h512.png" alt="Train set before calibration" title="Train set before calibration" width="300"/>
          </center>

          * **AFTER CALIBRATION**: 16,603 where 10,089 (60.77%) are in 0.2 <= p <= 0.5, and 6,514 (39.23%) are in p > 0.5.

          <center>
          <img src="./folds/camelyon16/w-512xh-512/patches-level-first-pass/training/AFTER-CALIBRATION-train-set-patch-w-512-h512.png" alt="Train set after calibration" title="Train set after calibration" width="300"/>
          </center>
        * **Original test WSI**:
          * **BEFORE CALIBRATION**: 61,442 where 4,823 (7.85%) are in 0.2 <= p <= 0.5, and 56,619 (92.15%) are in p > 0.5.

          <center>
          <img src="./folds/camelyon16/w-512xh-512/patches-level-first-pass/testing/BEFORE-CALIBRATION-test-set-patch-w-512-h512.png" alt="Test set before calibration" title="Test set before calibration" width="300"/>
          </center>

          * **AFTER CALIBRATION**: 7,832 where 4,823 (61.58%) are in 0.2 <= p <= 0.5, and 3,009 (38.42%) are in p > 0.5.

      <center>
      <img src="./folds/camelyon16/w-512xh-512/patches-level-first-pass/testing/AFTER-CALIBRATION-test-set-patch-w-512-h512.png" alt="Test set after calibration" title="Test set after calibration" width="300"/>
      </center>

   * #### Size `(w_=768, h_=768)`:
     * **Total patches**:
       * **BEFORE CALIBRATION**: 64,127 where 9,512 (14.83%) are in 0.2 <= p <= 0.5, and 54,615 (85.17%) are in p > 0.5.
       * **AFTER CALIBRATION**: 15,377 where 9,512 (61.86%) are in 0.2 <= p <= 0.5, and 5,865 (38.14%) are in p > 0.5.
     * **Original train WSI**:
       * **BEFORE CALIBRATION**: 35,791 where 6421 (17.94%) are in 0.2 <= p <= 0.5, and 29,370 (82.06%) are in p > 0.5.

       <center>
       <img src="./folds/camelyon16/w-768xh-768/patches-level-first-pass/training/BEFORE-CALIBRATION-train-set-patch-w-768-h768.png" alt="Train set before calibration" title="Train set before calibration" width="300"/>
       </center>

       * **AFTER CALIBRATION**: 10,437 where 6421 (61.52%) are in 0.2 <= p <= 0.5, and 4,016 (38.48%) are in p > 0.5.

       <center>
       <img src="./folds/camelyon16/w-768xh-768/patches-level-first-pass/training/AFTER-CALIBRATION-train-set-patch-w-768-h768.png" alt="Train set after calibration" title="Train set after calibration" width="300"/>
       </center>
     * **Original test WSI**:
       * **BEFORE CALIBRATION**: 28,336 where 3,091 (10.91%) are in 0.2 <= p <= 0.5, and 25,245 (89.09%) are in p > 0.5.

       <center>
       <img src="./folds/camelyon16/w-768xh-768/patches-level-first-pass/testing/BEFORE-CALIBRATION-test-set-patch-w-768-h768.png" alt="Test set before calibration" title="Test set before calibration" width="300"/>
       </center>

       * **AFTER CALIBRATION**: 4,940 where 3,091 (62.57%) are in 0.2 <= p <= 0.5, and 1,849 (37.43%) are in p > 0.5.

   <center>
   <img src="./folds/camelyon16/w-768xh-768/patches-level-first-pass/testing/AFTER-CALIBRATION-test-set-patch-w-768-h768.png" alt="Test set after calibration" title="Test set after calibration" width="300"/>
   </center>

   * #### Size `(w_=1024, h_=1024)`:
     * **Total patches**:
       * **BEFORE CALIBRATION**: 37,598 where 6,988 (18.59%) are in 0.2 <= p <= 0.5, and 30,610 (81.41%) are in p > 0.5.
       * **AFTER CALIBRATION**: 11,470 where 6988 (60.92%) are in 0.2 <= p <= 0.5, and 4,482 (39.08%) are in p > 0.5.
     * **Original train WSI**:
       * **BEFORE CALIBRATION**: 21,209 where 4,783 (22.55%) are in 0.2 <= p <= 0.5, and 16,426 (77.45%) are in p > 0.5.

       <center>
       <img src="./folds/camelyon16/w-1024xh-1024/patches-level-first-pass/training/BEFORE-CALIBRATION-train-set-patch-w-1024-h1024.png" alt="Train set before calibration" title="Train set before calibration" width="300"/>
       </center>

       * **AFTER CALIBRATION**: 7,815 where 4783 (61.20%) are in 0.2 <= p <= 0.5, and 3,032 (38.80%) are in p > 0.5.

       <center>
       <img src="./folds/camelyon16/w-1024xh-1024/patches-level-first-pass/training/AFTER-CALIBRATION-train-set-patch-w-1024-h1024.png" alt="Train set after calibration" title="Train set after calibration" width="300"/>
       </center>
     * **Original test WSI**:
       * **BEFORE CALIBRATION**: 16,389 where 2,205 (13.45%) are in 0.2 <= p <= 0.5, and 14,184 (86.55%) are in p > 0.5.

       <center>
       <img src="./folds/camelyon16/w-1024xh-1024/patches-level-first-pass/testing/BEFORE-CALIBRATION-test-set-patch-w-1024-h1024.png" alt="Test set before calibration" title="Test set before calibration" width="300"/>
       </center>

       * **AFTER CALIBRATION**: 3,655 where 2,205 (60.33%) are in 0.2 <= p <= 0.5, and 1,450 (39.67%) are in p > 0.5.

   <center>
   <img src="./folds/camelyon16/w-1024xh-1024/patches-level-first-pass/testing/AFTER-CALIBRATION-test-set-patch-w-1024-h1024.png" alt="Test set after calibration" title="Test set after calibration" width="300"/>
   </center>

12. All patches are sampled at `level=0` in the WSI (i.e., the highest resolution).
13. In a set (within a split, and fold), the number of normal patches sampled is equal (when possible) to the number of metastatic patches (after calibration).
14. Sampling metastatic patches is done ONLY ONCE. First, all the metastatic patches in `CAMELYON16` dataset are sampled. Then, they are calibrated. [original train, and test sets are treated separately].
15. Sampling normal patches is done for each split, each fold, each set (train, valid) for each WSI. Sampling twice [within the code] from the same WSI will provide different samples due to the randomness. (Due to the determinism of
the code, running the code twice will result in exactly the same samples.)
16. The test set of the fold is sampled ONLY ONCE. The same set is used across all the splits, folds.
17. Sampling normal patches is extremely fast, and extremely light in terms of memory. We used `48` processes with `1024MB` of memory for each to sample the entire normal patches for all the splits and folds. It took: `00:25:14 (hh:mm:ss)`.
18. Sampling time of metastatic and normal patches is practically independent of the size of the patch. In general, the total time of sampling sampling all metastatic, and normal patches for all splits, and folds over a server with `48` processes with `1024MB` each takes `03:41:00` `(hh:mm:00)`.
19. Some of the normal WSI are small; and they do not have enough tissue to sample `n` patches such as the case of `training/normal/normal_042.tif` with the size `(w_, h_) = (768, 768)` (in the case of patches without overlap). In this case, we sample what we can only.



### Patches files name system:
The path of patch/mask is relative.
1. Tag of a `metastatic` patch:
  * Example:
    `file_training-tumor-tumor_062.tif_reg_0_row_8_patch_0_x_84107_y_117472_tissue_0.3582255045572917_metastatic_0.2003716362847222_w_768_h_768`
  * Format:
  `file_{relative path of the WSI where the patch were sampled from}_reg_{number of the region, intern to the code}_row_{number of the row, intern to the code}_patch_{number of the patch in the WSI, intern to the code}_x_{the x of the upper left corner in the WSI}_y_{the y of the upper left corner in the WSI}_tissue_{percentage of the tissue pixels in th patch / 100}_metastatic_{percentage of the metastatic pixels in the patch / 100}_w_{the width of the patch}_h_{the hiegth of the pacth}`


  1. Name of the patch file:
    * Example: `Patch_file_training-tumor-tumor_062.tif_reg_0_row_8_patch_0_x_84107_y_117472_tissue_0.3582255045572917_metastatic_0.2003716362847222_w_768_h_768.png`
    * Format: `Patch_{tag of the path}.png`

  2. Name of the path mask:
    * Example: `mask_file_training-tumor-tumor_062.tif_reg_0_row_8_patch_0_x_84107_y_117472_tissue_0.3582255045572917_metastatic_0.2003716362847222_w_768_h_768.png`
    * Format: `mask_{tag of the path}.png`



2. Tag of a `normal` patch:
  * Example:
    `file_training-normal-normal_084.tif_patch_0_x_35840_y_94720_tissue_0.36586761474609375_w_512_h_512`
  * Format:
  `file_{relative path of the WSI where the patch were sampled from}_patch_{number of the patch in the WSI, intern to the code}_x_{the x of the upper left corner in the WSI}_y_{the y of the upper left corner in the WSI}_tissue_{percentage of the tissue pixels in th patch / 100}_w_{the width of the patch}_h_{the hiegth of the pacth}`


##### Name of the patch file:
* Example: `Patch_file_training-normal-normal_084.tif_patch_0_x_35840_y_94720_tissue_0.36586761474609375_w_512_h_512.png`
* Format: `Patch_{tag of the path}.png`


### Used configuration for sampling:

See `./wsi.py`, function `run()`.

We sampled three sizes of patches `(w, h)`: `[[512, 512], [768, 768], [1024, 1024]]`.

```python

def do_one_size(current_w, current_h):
    """
    Do all the sampling: Pass 1, 2, 3, 4 for a specific size.

    :param current_w: int, width of the pacth.
    :param current_h: int, height of the patch.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    t0 = dt.datetime.now()
    # Size of the patches.

    h_ = current_h
    w_ = current_w
    delta_w = w_  # no overlap.
    delta_h = h_  # no overlap.

    outd_first = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-first-pass"))
    outd_second = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-second-pass"))
    outd_third = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-third-pass"))
    outd_fourth = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-fourth-pass"))

    args = {"level": 0,  # level of resolution in the WSI. 0 is the highest resolution [Juts for debug]
            "level_approx": 6,  # resolution level where the search for the patches is done.
            "rgb_min": 10,  # minimal value of RGB color. Used in thresholding to estimate the tissue mask.
            "delta_w": delta_w,  # horizontal stride of the sliding window when sampling patches.
            "delta_h": delta_h,  # vertical stride of the sliding window when sampling patches.
            "h_": h_,  # height of the patch.
            "w_": w_,  # width of the patch.
            "tissue_mask_min": 0.1,  # minimal percentage of tissue mask in a patch to be accepted.
            "p0": 0.2,  # p0
            "p1": 0.5,  # p1
            "delta_w_inter": 1,  # horizontal stride of the sliding window when sampling patches. Used to approach
            # SLOWLY the border of the tumor. It is better to keep it to 1.
            "dichotomy_step": 100,  # NO LONGER USEFUL. # TODO: REMOVE IT.
            "path_wsi": None,  # path to the WSI.
            "path_xml": None,  # path to the xml file of the annotation of the WSI.
            "debug": False,  # USED FOR DEBUG. NO LONGER USEFUL. TODO: REMOVE IT.
            "outd_patches": join(baseurl, join("w-{}xh-{}".format(w_, h_), "metastatic-patches")),  # path where the
            # patches will be saved.
            "outd": outd_first,  # path where the *.csv files of the first pass will be saved.
            "outd_second": outd_second,  # path to directory where *.csv files of the second pass will be stored.
            "outd_third": outd_third,  # path to directory where *.csv files of the third pass will be stored.
            "outd_fourth": outd_fourth,  # path to directory where *.csv files of the fourth (final) pass are
            # stored.
            "fold": "./folds/camelyon16/WSI-level/split_0/fold_0",  # use a random split/fold.
            "n": 0.01,  # a percentage (of the patches with p0 <= p <= p1) used to compute the number of patches
            # with p > p1 that we should consider. This number is computed as: N * n, where N is the number of
            # patches with p0 <= p <= p1.
            "n_norm": None,  # number of normal patches to sample.
            "splits_dir": "folds/camelyon16/WSI-level",  # directory where the splits are.
            "baseurl": baseurl,  # the base path to CAMELYON16 dataset.
            "delta_bin": 0.05,  # delta used to create bins for sampling (calibrate metastatic patches).
            "max_trials_inside_bin": 100,  # maximum number of times to try sampling a sample from a bin if the
            # sampling fails because we sampled all the samples inside that bin.
            "max_trials_sample_normal_patches_inside_wsi": 1000,  # maximum number to try to sample a normal patch
            # within a WSI before giving up and moving to the next patch. Some WSI contain very small region of
            # tissue, and it makes it difficult to sample in the case of non-overlapping patches.
            }

    sample_all_metastatic_patches_and_calibrate_them(ind, outd_first, outd_second, nbr_workers, baseurl, args)

    # Do the splits/folds
    # absolute path where the normal patches will be saved.
    args["outd_patches"] = join(baseurl, join("w-{}xh-{}".format(w_, h_), "normal-patches"))
    # relative path where the metastatic patches have been saved.
    args["relative_path_metastatic_patches"] = join("w-{}xh-{}".format(w_, h_), "metastatic-patches")
    do_splits_folds(args, nbr_workers)
```


## How to run the code?
The code is made to create the splits/k-folds (and sample from `CAMELYON16` dataset) for the experiments.

1. To create the splits/k-folds of all the datasets use:
```sh
$ python create_folds.py
```
The script will output in the folder `./folds`.

2. To sample from `CAMELYON16` dataset:
    1. First, we need to create the splits/k-folds in WSI-level:
    ```sh
    $ python create_folds.py
    ```

    2. Then, sample the patches (use multi-processing to speedup). Example with `10` processes:
    ```sh
    $ python wsi.py 10
    ```
    The `*.csv` files will be stored in `./folds` while the patches and their corresponding masks will be stored in the the directory where `CAMELYON16` dataset is stored.
