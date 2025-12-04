import os
import random
import shutil
from typing import Dict, List

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

    def load_mask(self, mask_path: str) -> np.array:
        """
        Loads a mask from the specified path.

        Parameters:
        mask_path (str): The path to the mask file.

        Returns:
        np.array: The loaded mask.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        return mask

    def extract_patches(
        self,
        image_path: str,
        mask_path: str = None,
        skip_background: bool = True,
        skip_no_shoreline: int = False,
        binary_class: bool = False,
        padding_mode: str = "constant",
        p_keep_negative: int = 0,
    ) -> List[dict]:
        """
        Extracts patches from the image and mask located at the given paths.

        Parameters:
        image_path (str): The path to the image file.
        mask_path (str): The path to the mask file. Default: None
        skip_background (bool): Whether to skip patches that do not contain any kind of pixel. Default: True
        skip_no_shoreline (int): The value of the pixel indicating the shoreline. If the patch does not contain this value, it will be skipped. Default: None
        padding_mode (str): The padding mode to use, either 'constant' or 'reflect'. Default: 'constant'

        Returns:
        dict: A dictionary containing the extracted patches and padding information.
        """
        # Load the image and mask
        image = self.load_image(image_path)
        if mask_path is not None:
            mask = self.load_mask(mask_path)

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

        if mask_path:
            # Pad the mask similarly based on the padding_mode
            if padding_mode == "reflect":
                padded_mask = np.pad(
                    mask,
                    ((padding_top, padding_bottom), (padding_left, padding_right)),
                    mode="reflect",
                )
            else:
                padded_mask = np.pad(
                    mask,
                    ((padding_top, padding_bottom), (padding_left, padding_right)),
                    mode="constant",
                    constant_values=0,
                )

            patches_mask = patchify(
                padded_mask,
                (self.patch_size[0], self.patch_size[1]),
                step=(self.stride[0], self.stride[1]),
            )

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

                if mask_path:
                    mask_patch = patches_mask[i, j, :, :]
                    if (
                        skip_background and np.sum(mask_patch) == 0
                    ):  # Skip patch if mask has no pixels
                        continue

                    # Skip patch if mask has no skip_no_shoreline
                    if skip_no_shoreline:
                        if binary_class:
                            has_class_1 = np.any(mask_patch == 0)
                            has_class_2 = np.any(mask_patch == 1)
                        else:
                            has_class_1 = np.any(mask_patch == 1)
                            has_class_2 = np.any(mask_patch == 2)

                        if has_class_1 and has_class_2:
                            pass
                        elif random.random() > p_keep_negative:
                            continue

                    patch_info["mask"] = mask_patch
                    base_name, ext = os.path.splitext(mask_path)
                    patch_info["mask_path"] = (
                        f"{os.path.basename(base_name)}.patch.{i}_{j}{ext}"
                    )

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
        mask_path: str = None,
        output_image_dir: str = "data/patchify/train/images",
        output_mask_dir: str = "data/patchify/train/masks",
        skip_background: bool = False,
        skip_no_shoreline: bool = False,
        padding_mode: str = "constant",
        binary_class: bool = False,
        p_keep_negative: int = 0,
    ) -> None:
        """
        Extracts patches from the image and mask, and saves them to the specified directory.

        Parameters:
        image_path (str): The path to the image file.
        mask_path (str): The path to the mask file. Default: None.
        output_image_dir (str): The directory where the image patches will be saved. Default: 'data/patchify/train/images'.
        output_mask_dir (str): The directory where the mask patches will be saved. Default: 'data/patchify/train/masks'.
        skip_no_shoreline (int): The value of the pixel indicating the shoreline. If the patch does not contain this value, it will be skipped. Default: None
        padding_mode (str): The padding mode to use, either 'constant' or 'reflect'. Default: 'constant'

        Returns:
        dict: A dictionary containing the extracted patches and padding information.
        """
        result = self.extract_patches(
            image_path,
            mask_path,
            padding_mode=padding_mode,
            skip_no_shoreline=skip_no_shoreline,
            binary_class=binary_class,
            skip_background=skip_background,
            p_keep_negative=p_keep_negative,
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

    def extract_patches_and_save(
        self,
        data: Dict[str, Dict[str, List]],
        output_dir: str = "data/patchify/",
        skip_no_shoreline: int = None,
        padding_mode: str = "constant",
        skip_background: bool = False,
        binary_class: bool = False,
        p_keep_negative: int = 0,
    ) -> None:
        """
        Extract patches from images and masks, and save them into the specified directory
        for training, validation, and testing datasets.

        Parameters:
        data (Dict[str, Dict[str, List]]): A dictionary containing the patches for each dataset (train, val, test).
        output_dir (str): The directory where the patches will be saved.
        skip_no_shoreline (int): The value of the pixel indicating the shoreline. If the patch does not contain this value, it will be skipped. Default: None
        padding_mode (str): The padding mode to use, either 'constant' or 'reflect'. Default: 'constant'
        """

        # Remove the output directory if it already exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        for dataset, dataset_data in data.items():
            print(f"Extracting patches for {dataset} dataset...")
            dataset_dir = os.path.join(output_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            # Create the X and y directories
            x_dir = os.path.join(dataset_dir, "images")
            y_dir = os.path.join(dataset_dir, "masks")

            os.makedirs(x_dir, exist_ok=True)
            os.makedirs(y_dir, exist_ok=True)

            for i, (image_path, mask_path) in enumerate(
                zip(dataset_data["images"], dataset_data["masks"])
            ):
                self.extract_an_image_and_save_patches(
                    image_path,
                    mask_path,
                    x_dir,
                    y_dir,
                    padding_mode=padding_mode,
                    skip_no_shoreline=skip_no_shoreline,
                    skip_background=skip_background,
                    binary_class=binary_class,
                    p_keep_negative=p_keep_negative,
                )

            print(f"Finished extracting patches for {dataset} dataset.\n")

    def save_patch(self, patch: np.array, patch_path: str, patch_name: str) -> None:
        """
        Saves a patch to the specified directory.

        Parameters:
        patch (np.array): The patch to be saved.
        patch_path (str): The name of the patch folder.
        patch_name (str): The name of the patch file.
        """
        cv2.imwrite(os.path.join(patch_path, patch_name), patch)
