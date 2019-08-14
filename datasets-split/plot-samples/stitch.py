from PIL import Image
import os
from os.path import join
import glob


# ==============================================
#                    DRAW
# ==============================================

def stitch_it(dirin):
    """
    Stitch images.

    :param dirin:
    :return:
    """
    delta = 4  # pixels horizontal and vertical space.

    normal = join(dirin, "normal")
    tumor = join(dirin, "tumor")

    nbr = 4  # nbr_images per class
    k = 2  # nb rows
    h, w = [int(t) for t in dirin.split(os.sep)[-1].split("x")]

    tumor_patches = sorted([x for x in glob.glob(join(tumor, "*.png")) if x.split(os.sep)[-1].startswith("Patch_")])
    tumor_masks = sorted([x for x in glob.glob(join(tumor, "*.png")) if x.split(os.sep)[-1].startswith("mask_")])

    img_tumor = Image.new("RGB", (w * nbr + (nbr - 1) * delta, h * k + (nbr - 1) * delta))

    for j, patch in enumerate(tumor_patches):
        tmp = Image.open(patch, "r").convert("RGB")
        img_tumor.paste(tmp, (j * (w + delta), 0), None)

    for j, mask in enumerate(tumor_masks):
        tmp = Image.open(mask, "r").convert("RGB")
        img_tumor.paste(tmp, (j * (w + delta), h), None)

    img_tumor.save(join(dirin, "tumor.jpeg"), "JPEG")

    normal_patches = sorted([x for x in glob.glob(join(normal, "*.png")) if x.split(os.sep)[-1].startswith("Patch_")])

    img_normal = Image.new("RGB", (w * nbr, h))
    for j, patch in enumerate(normal_patches):
        tmp = Image.open(patch, "r").convert("RGB")
        img_normal.paste(tmp, (j * (w + delta), 0), None)

    img_normal.save(join(dirin, "normal.jpeg"), "JPEG")


if __name__ == "__main__":
    stitch_it("512x512")
    stitch_it("768x768")
    stitch_it("1024x1024")

