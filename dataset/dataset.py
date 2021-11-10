import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import os
import albumentations as A
from matplotlib.pyplot import get_cmap


class RailData(Dataset):
    def __init__(
        self,
        images_path,
        mask_path,
        res_scale,
        pix_scale="min_max",
        transform=False
    ):
        """
        Initialize Dataset.

        :param images_path: Path to images.
        :param mask_path: Path to segmentation masks.
        :param res_scale: Scale image and mask height and width.
        :param pix_scale: Scale of the pixels. Ether 'min_max' or 'std'.
        :return: None.
        """
        self._res_scale = res_scale
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
        if 1 <= res_scale < 0:
            err = f"Scale factor must be between 0 and 1, got {res_scale}"
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
        width, height = image_img.size
        new_width, new_height = width * self._res_scale, height * self._res_scale
        new_width, new_height = round(new_width), round(new_height)
        # image_img = image_img.resize((new_width, new_height))
        # mask_img = mask_img.resize((new_width, new_height))
        image_img = image_img.resize((160, 320))
        mask_img = mask_img.resize((160, 320))

        """
        # List transformations
        transform = A.Compose(
            [
                A.RandomResizedCrop(height=new_height, width=new_width, p=0.9, scale=(0.8, 1.0,)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=(-2, 2,), p=0.9)
            ]
        )
        """

        # Convert images and mask to array
        image_arr = np.asarray(image_img).copy()
        mask_arr = np.asarray(mask_img).copy()

        # Make sure only 2 classes
        highest_class = np.max(mask_arr)
        lowest_class = np.min(mask_arr)
        if highest_class > 1 or lowest_class < 0:
            classes = np.unique(mask_arr)
            err = f"Expected two classes [0 1], got {len(classes)}: {classes}."
            raise ValueError(err)

        # Augment Images
        """
        if self.transform:
            augmentations = transform(image=image_arr, mask=mask_arr)
            image_arr = augmentations["image"]
            mask_arr = augmentations["mask"]
        """

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
        # print(f"{image_arr.shape=}")
        # print(f"{mask_arr.shape=}")

        return {
            "image": torch.from_numpy(img_trans).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask_arr).type(torch.FloatTensor),
        }
