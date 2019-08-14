# `*.csv` file format:
absolute path to the patch image, absolute path to the mask image (if the class is `tumor`), class (`normal` or `tumor`).


# Tag of a `metastatic` patch:
* Example:
  `file_training-tumor-tumor_062.tif_reg_0_row_8_patch_0_x_84107_y_117472_tissue_0.3582255045572917_metastatic_0.2003716362847222_w_768_h_768`
* Format:
`file_{relative path of the WSI where the patch were sampled from}_reg_{number of the region, intern to the code}_row_{number of the row, intern to the code}_patch_{number of the patch in the WSI, intern to the code}_x_{the x of the upper left corner in the WSI}_y_{the y of the upper left corner in the WSI}_tissue_{percentage of the tissue pixels in th patch / 100}_metastatic_{percentage of the metastatic pixels in the patch / 100}_w_{the width of the patch}_h_{the hiegth of the pacth}`


## Name of the patch file:
* Example: `Patch_file_training-tumor-tumor_062.tif_reg_0_row_8_patch_0_x_84107_y_117472_tissue_0.3582255045572917_metastatic_0.2003716362847222_w_768_h_768.png`
* Format: `Patch_{tag of the path}.png`

## Name of the path mask:
* Example: `mask_file_training-tumor-tumor_062.tif_reg_0_row_8_patch_0_x_84107_y_117472_tissue_0.3582255045572917_metastatic_0.2003716362847222_w_768_h_768.png`
* Format: `mask_{tag of the path}.png`



## Tag of a `normal` patch:
* Example:
  `file_training-normal-normal_084.tif_patch_0_x_35840_y_94720_tissue_0.36586761474609375_w_512_h_512`
* Format:
`file_{relative path of the WSI where the patch were sampled from}_patch_{number of the patch in the WSI, intern to the code}_x_{the x of the upper left corner in the WSI}_y_{the y of the upper left corner in the WSI}_tissue_{percentage of the tissue pixels in th patch / 100}_w_{the width of the patch}_h_{the hiegth of the pacth}`


## Name of the patch file:
* Example: `Patch_file_training-normal-normal_084.tif_patch_0_x_35840_y_94720_tissue_0.36586761474609375_w_512_h_512.png`
* Format: `Patch_{tag of the path}.png`
