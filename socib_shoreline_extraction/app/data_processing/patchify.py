import os
import random
from typing import List

import cv2
import numpy as np
from patchify import patchify


class Patchify:
    def __init__(self, patch_size: tuple = (256, 256), stride: tuple = (128, 128)):
        """
        Initializes the Patchify object.

        Parameters:
        patch_size (tuple): The size of each patch with (height, width). Default: (256, 256).
        stride (tuple): The stride (step size) for moving the window with (vertical, horizontal). Default: (128, 128).
        """
        self.patch_size = patch_size
        self.stride = stride

        random.seed(42)  # Set a random seed for reproducibility
        np.random.seed(42)  # Set a random seed for reproducibility

    def load_image(self, image_path: str) -> np.array:
        """
        Loads an image from the specified path.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        np.array: The loaded image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return image

    def extract_patches(
        self,
        image_path: str,
        skip_background: bool = True,
        skip_no_shoreline: int = False,
        binary_class: bool = False,
        padding_mode: str = "constant",
    ) -> List[dict]:
        """
        Extracts patches from the image and mask located at the given paths.

        Parameters:
        image_path (str): The path to the image file.
        skip_background (bool): Whether to skip patches that do not contain any kind of pixel. Default: True
        skip_no_shoreline (int): The value of the pixel indicating the shoreline. If the patch does not contain this value, it will be skipped. Default: None
        padding_mode (str): The padding mode to use, either 'constant' or 'reflect'. Default: 'constant'

        Returns:
        dict: A dictionary containing the extracted patches and padding information.
        """
        # Load the image and mask
        image = self.load_image(image_path)

        # Get the image dimensions
        height, width, _ = image.shape  # height, width, channels

        # Calculate the padding needed to make the image divisible by patch_size
        aux_height = height % self.patch_size[0]
        aux_width = width % self.patch_size[1]

        aux_height = self.patch_size[0] - aux_height
        aux_width = self.patch_size[1] - aux_width

        # Padding for the image to be divisible by patch_size
        padding_top = aux_height // 2
        padding_bottom = aux_height // 2
        if aux_height % 2 != 0:
            padding_bottom += 1

        padding_left = aux_width // 2
        padding_right = aux_width // 2
        if aux_width % 2 != 0:
            padding_right += 1

        # Pad the image
        # padded_image = np.pad(image, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='constant', constant_values=0)

        # Pad the image based on the padding_mode
        if padding_mode == "reflect":
            padded_image = np.pad(
                image,
                ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                mode="reflect",
            )
        else:
            padded_image = np.pad(
                image,
                ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        patches_img = patchify(
            padded_image,
            (self.patch_size[0], self.patch_size[1], 3),
            step=(self.stride[0], self.stride[1], 3),
        )

        patches = []

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                img_patch = patches_img[i, j, 0, :, :, :]
                base_name, ext = os.path.splitext(image_path)
                patch_info = {
                    "row": i,
                    "col": j,
                    "image": img_patch,
                    "image_path": f"{os.path.basename(base_name)}.patch.{i}_{j}{ext}",
                }

                patches.append(patch_info)

        return {
            "patches": patches,
            "padding": {
                "top": padding_top,
                "bottom": padding_bottom,
                "left": padding_left,
                "right": padding_right,
            },
            "options": {
                "size": self.patch_size,
                "stride": self.stride,
            },
        }

    def extract_an_image_and_save_patches(
        self,
        image_path: str,
        output_image_dir: str = "data/patchify/train/images",
        output_mask_dir: str = "data/patchify/train/masks",
        skip_background: bool = False,
        skip_no_shoreline: bool = False,
        padding_mode: str = "constant",
        binary_class: bool = False,
    ) -> None:
        """
        Extracts patches from the image and mask, and saves them to the specified directory.

        Parameters:
        image_path (str): The path to the image file.
        output_image_dir (str): The directory where the image patches will be saved. Default: 'data/patchify/train/images'.
        output_mask_dir (str): The directory where the mask patches will be saved. Default: 'data/patchify/train/masks'.
        skip_no_shoreline (int): The value of the pixel indicating the shoreline. If the patch does not contain this value, it will be skipped. Default: None
        padding_mode (str): The padding mode to use, either 'constant' or 'reflect'. Default: 'constant'

        Returns:
        dict: A dictionary containing the extracted patches and padding information.
        """
        result = self.extract_patches(
            image_path,
            padding_mode=padding_mode,
            skip_no_shoreline=skip_no_shoreline,
            binary_class=binary_class,
            skip_background=skip_background,
        )
        patches = result["patches"]

        # Iterate over patches and save them
        for i, patch in enumerate(patches):
            patch_name = patch["image_path"]
            patch_image = patch["image"]
            self.save_patch(patch_image, output_image_dir, patch_name)

            if "mask" in patch:
                mask_name = patch["mask_path"]
                mask_image = patch["mask"]
                self.save_patch(mask_image, output_mask_dir, mask_name)

        return result

    def save_patch(self, patch: np.array, patch_path: str, patch_name: str) -> None:
        """
        Saves a patch to the specified directory.

        Parameters:
        patch (np.array): The patch to be saved.
        patch_path (str): The name of the patch folder.
        patch_name (str): The name of the patch file.
        """
        cv2.imwrite(os.path.join(patch_path, patch_name), patch)
