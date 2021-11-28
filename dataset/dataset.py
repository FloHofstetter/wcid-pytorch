import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import os
import albumentations as A
from matplotlib.pyplot import get_cmap
from typing import List, Tuple, Dict


class RailData(Dataset):
    def __init__(
        self,
        images_path: str,
        mask_path: str,
        resolution: Tuple[int, int] = (320, 160),
        pix_scale="min_max",
        transform=False,
    ):
        """
        Initialize Dataset.

        :param images_path: Path to images.
        :param mask_path: Path to segmentation masks.
        :param res_scale: Scale image and mask height and width.
        :param pix_scale: Scale of the pixels. Ether 'min_max' or 'std'.
        :return: None.
        """
        self.resolution = resolution
        self._pix_scale = pix_scale
        self.transform = transform

        # Collect images paths
        images_path = os.path.join(images_path, "*.png")
        image_paths = glob.glob(images_path)
        self._image_paths = sorted(image_paths)

        # Collect mask paths
        masks_path = os.path.join(mask_path, "*.png")
        mask_paths = glob.glob(masks_path)
        self._mask_paths = sorted(mask_paths)

        # Sanity checks
        if self.resolution[0] < 1 or self.resolution[1] < 1:
            err = f"Resolution mus be grater than 1, got {self.resolution}"
            raise ValueError(err)

        if not len(self._image_paths) == len(self._mask_paths):
            err = (
                f"Amount of images and masks must be the same,"
                + f"got {len(self._image_paths)} and"
                + f" {len(self._mask_paths)} masks"
            )
            raise ValueError(err)

    def __len__(self):
        """
        Get length of the elements in dataset.

        :return: None.
        """
        return len(self._image_paths)

    def __getitem__(self, index):
        """
        Get next element of Dataset.

        :param index: Number of element in dataset.
        :return: Element in dataset.
        """
        # Open images and masks
        image_img = Image.open(self._image_paths[index])
        mask_img = Image.open(self._mask_paths[index])

        # Resize images and masks
        image_img = image_img.resize(
            size=(self.resolution[1], self.resolution[0]), resample=Image.BICUBIC
        )
        mask_img = mask_img.resize(
            size=(self.resolution[1], self.resolution[0]), resample=Image.BICUBIC
        )

        # Convert images and mask to array
        image_arr = np.asarray(image_img).copy()
        mask_arr = np.asarray(mask_img).copy()

        # Make sure only 2 classes
        highest_class = np.max(mask_arr)
        lowest_class = np.min(mask_arr)
        if highest_class > 1 or lowest_class < 0:
            # join all non zero label together if more than 2 classes are present
            mask_arr = mask_arr.astype(bool).astype(np.uint8)

        # List transformations
        transform = A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.resolution[1],
                    width=self.resolution[0],
                    p=0.0,
                    scale=(0.8, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=(-2.5, 2.5), p=0.9),
                A.MotionBlur(
                    always_apply=False,
                    p=0.1,
                    blur_limit=(14, 20),
                ),
            ]
        )

        if self.transform:
            # Albumentations augmentation
            augmentations = transform(image=image_arr, mask=mask_arr)
            image_arr = augmentations["image"]
            mask_arr = augmentations["mask"]

        # Expand Mask dimenseion
        mask_arr = np.expand_dims(mask_arr, axis=0)

        # Cast datatype and normalize Image
        image_arr = image_arr.astype(np.float32)
        mask_arr = mask_arr.astype(np.float32)

        # Scale images
        if self._pix_scale == "min_max":
            image_arr /= 255
        elif self._pix_scale == "std":
            mean = np.asarray([157.03, 96.86, 81.24])
            std = np.asarray([60.68, 53.79, 36.77])
            image_arr = (image_arr - mean) / std
        elif self._pix_scale == "indi_std":
            image1_arr = image_arr.reshape((3, -1))
            mean = np.mean(image1_arr, axis=1)
            std = np.std(image1_arr, axis=1)
            std += np.finfo(np.float64).eps
            image_arr = (image_arr - mean) / std
        else:
            err = "Expected parameter pix_scale to be 'min_max' or 'std', got"
            err += f" {self._pix_scale}"
            raise ValueError()

        # Height width channels to channel height width
        img_trans = image_arr.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(img_trans).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask_arr).type(torch.FloatTensor),
        }
