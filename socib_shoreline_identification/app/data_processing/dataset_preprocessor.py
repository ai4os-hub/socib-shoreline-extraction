import os
import cv2
import numpy as np
import shutil
import json

from typing import Tuple

class DatasetPreprocessor:
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
    
    def get_rows_with_class(self, img: np.array, mask: np.array, type_class: int = 255) -> Tuple[np.array, np.array]:
        """
        Returns the rows that contain the specified class. 

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        type_class (int): The color class to search for. Default: 255

        Returns:
        Tuple[np.array, np.array]: The new image and mask with only the selected rows.
        """

        rows = np.where(np.any(mask == type_class, axis=1))[0]
        
        new_img = img[rows, :]
        new_mask = mask[rows, :]

        return new_img, new_mask
    
    def transform_class_to_background(self, img: np.array, mask: np.array, type_class: int = 255, background_class: int = 0) -> Tuple[np.array, np.array]:
        """
        Transforms the specified class to the background class.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        type_class (int): The color class to transform. Default: 255
        background_class (int): The background class. Default: 0

        Returns:
        Tuple[np.array, np.array]: The new image and mask with the transformed class.
        """
        img[mask == type_class] = [background_class, background_class, background_class]
        mask[mask == type_class] = background_class

        return img, mask
    
    def remove_rows_with_background(self, img: np.array, mask: np.array, background_class: int = 0) -> Tuple[np.array, np.array]:
        """
        Removes the rows that contain only the background class.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        background_class (int): The background class. Default: 0

        Returns:
        Tuple[np.array, np.array]: The new image and mask with only the selected rows.
        """

        rows = np.where(np.any(mask != background_class, axis=1))[0]

        new_img = img[rows, :]
        new_mask = mask[rows, :]

        return new_img, new_mask
    
    def remove_cols_with_background(self, img: np.array, mask: np.array, background_class: int = 0) -> Tuple[np.array, np.array]:
        """
        Removes the columns that contain only the background class.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        background_class (int): The background class. Default: 0

        Returns:
        Tuple[np.array, np.array]: The new image and mask with only the selected columns.
        """

        cols = np.where(np.any(mask != background_class, axis=0))[0]

        new_img = img[:, cols]
        new_mask = mask[:, cols]

        return new_img, new_mask
    
    def remove_rows_with_some_background(self, img: np.array, mask: np.array, background_class: int = 0) -> Tuple[np.array, np.array]:
        """
        Removes the rows that contain only the background class.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        background_class (int): The background class. Default: 0

        Returns:
        Tuple[np.array, np.array]: The new image and mask with only the selected rows.
        """
        n_rows = mask.shape[0]
        rows = []

        total_pixels = mask.shape[1]

        for row in range(n_rows):
            background_pixels = np.sum(mask[row, :] == background_class)
            if background_pixels > total_pixels*0.50:  # If more than 50% of the pixels are background, skip this row
                continue

            rows.append(row)

        new_img = img[rows, :]
        new_mask = mask[rows, :]

        return new_img, new_mask
    
    def remove_cols_with_some_background(self, img: np.array, mask: np.array, background_class: int = 0) -> Tuple[np.array, np.array]:
        """
        Removes the columns that contain only the background class.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        background_class (int): The background class. Default: 0

        Returns:
        Tuple[np.array, np.array]: The new image and mask with only the selected columns.
        """
        n_cols = mask.shape[1]
        cols = []

        total_pixels = mask.shape[0]
        for col in range(n_cols):
            background_pixels = np.sum(mask[:, col] == background_class)
            if background_pixels > total_pixels*0.01:  # If more than 1% of the pixels are background, skip this column
                continue
            cols.append(col)

        new_img = img[:, cols]
        new_mask = mask[:, cols]

        return new_img, new_mask

    def mask_mapping(self, mask: np.array, mapping: dict) -> np.array:
        """
        Maps the mask classes to the new classes.

        Parameters:
        mask (np.array): The mask image.
        mapping (dict): The mapping of the classes. The key is the old class and the value is the new class.

        Returns:
        np.array: The new mask image.
        """
        if mapping is None:
            return mask

        new_mask = np.zeros_like(mask)

        for old_class, new_class in mapping.items():
            new_mask[mask == old_class] = new_class

        return new_mask
    
    def remove_rows_with_background_and_shoreline(self, img: np.array, mask: np.array, background_class: int = 0, shoreline_class: int = 255) -> Tuple[np.array, np.array]:
        """
        Removes the rows that contain only the background class or only the shoreline class.
        Keeps rows where the shoreline class is present along with other classes.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        background_class (int): The background class. Default: 0
        shoreline_class (int): The shoreline class. Default: 255

        Returns:
        Tuple[np.array, np.array]: The new image and mask with only the selected rows.
        """
        # Find rows where the shoreline class is present and other classes (not background) are also present
        rows = np.where(
            (np.any(mask == shoreline_class, axis=1)) &  # Rows with shoreline
            (np.any((mask != background_class) & (mask != shoreline_class), axis=1))  # Rows with other classes
        )[0]

        # Filter the image and mask to keep only the selected rows
        new_img = img[rows, :]
        new_mask = mask[rows, :]

        return new_img, new_mask
    
    def process_image(self, img: np.array, mask: np.array, type_class: int = 255, background_class: int = 0, mask_mapping: dict = None) -> Tuple[np.array, np.array]:
        """
        Processes the image and mask.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        type_class (int): The color class to transform. Default: 255
        background_class (int): The background class. Default: 0
        mask_mapping (dict): The mapping of the classes. The key is the old class and the value is the new class. Default: None

        Returns:
        Tuple[np.array, np.array]: The new image and mask.
        """
        img, mask = self.get_rows_with_class(img, mask, type_class)
        img, mask = self.transform_class_to_background(img, mask, type_class = 25, background_class = background_class) # 25 is the class for the not classified pixels
        img, mask = self.remove_rows_with_background(img, mask, background_class)
        img, mask = self.remove_cols_with_background(img, mask, background_class)
        img, mask = self.remove_rows_with_background_and_shoreline(img, mask, background_class, shoreline_class=type_class) # 255 is the class for the shoreline
        mask = self.mask_mapping(mask, mask_mapping) # 25 is the class for the not classified pixels

        return img, mask

    def process_image_oblique(self, img: np.array, mask: np.array, shoreline_class_pixel: int = 255, background_class: int = 0, mask_mapping: dict = None) -> Tuple[np.array, np.array]:
        """
        Processes the image and mask for oblique images.

        Parameters:
        img (np.array): The image.
        mask (np.array): The mask image.
        shoreline_class_pixel (int): The color class for the shoreline pixels. Default: 255
        background_class (int): The background class. Default: 0
        mask_mapping (dict): The mapping of the classes. The key is the old class and the value is the new class. Default: None

        Returns:
        Tuple[np.array, np.array]: The new image and mask.
        """
        img, mask = self.remove_rows_with_some_background(img, mask, background_class = 25)
        img, mask = self.remove_rows_with_some_background(img, mask, background_class = 0)
        img, mask = self.remove_cols_with_some_background(img, mask, background_class)
        mask = self.mask_mapping(mask, mask_mapping) # 25 is the class for the not classified pixels
        return img, mask

    def preprocess(self, dataset_path: str, dataset_output_path: str, mask_mapping: dict = None, oblique: bool = False) -> None:
        """
        Preprocesses the dataset located at the given path.
        The dataset should be organized in the following way:

        /dataset_path 
        ├── images
        │   ├── image_0.jpg
        │   ├── image_1.jpg
        │   └── ...
        ├── masks
        │   ├── mask_0.jpg
        │   ├── mask_1.jpg
        │   └── ...
        └── metadata.json

        Parameters:
        dataset_path (str): The path to the dataset.
        dataset_output_path (str): The path to save the preprocessed dataset.
        mask_mapping (dict): The mapping of the classes. The key is the old class and the value is the new class. Default: None
        oblique (bool): Whether the images are oblique or not. Default: False

        Returns:
        None
        """

        # Remove the output directory if it already exists
        if os.path.exists(dataset_output_path):
            shutil.rmtree(dataset_output_path)
        
        # Create the output directories
        output_images_path = os.path.join(dataset_output_path, "images")
        output_masks_path = os.path.join(dataset_output_path, "masks")
        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_masks_path, exist_ok=True)

        # Copy the metadata file
        metadata_path = os.path.join(dataset_path, "metadata.json")
        metadata_output_path = os.path.join(dataset_output_path, "metadata.json")
        if os.path.exists(metadata_path):
            shutil.copy(metadata_path, metadata_output_path)


        image_folder_paths = sorted(os.listdir(os.path.join(dataset_path, "images")))
        mask_folder_paths = sorted(os.listdir(os.path.join(dataset_path, "masks")))

        for i, (image_folder, mask_folder) in enumerate(zip(image_folder_paths, mask_folder_paths)):
            image_folder_path = os.path.join(dataset_path, "images", image_folder)
            mask_folder_path = os.path.join(dataset_path, "masks", mask_folder)

            image = self.load_image(image_folder_path)
            mask = self.load_mask(mask_folder_path)

            if oblique:
                image, mask = self.process_image_oblique(image, mask, mask_mapping=mask_mapping)
            else:
                image, mask = self.process_image(image, mask, mask_mapping=mask_mapping)

            output_image_path = os.path.join(output_images_path, image_folder)
            output_mask_path = os.path.join(output_masks_path, mask_folder)

            cv2.imwrite(output_image_path, image)
            cv2.imwrite(output_mask_path, mask)

    def preprocess_from_metadata(self, metadata: list, dataset_path: str, dataset_output_path: str, mask_mapping: dict = None) -> None:
        """
        Preprocesses the dataset using the metadata provided.

        Parameters:
        metadata (list): A list of dictionaries containing the metadata for each image and mask.
                        Each dictionary should contain the keys 'image_path' and 'mask_path'.
        dataset_path (str): The path to the dataset.
        dataset_output_path (str): The path to save the preprocessed dataset.
        mask_mapping (dict): The mapping of the classes. The key is the old class and the value is the new class. Default: None

        Returns:
        None
        """

        # Remove the output directory if it already exists
        if os.path.exists(dataset_output_path):
            shutil.rmtree(dataset_output_path)
        
        # Create the output directories
        output_images_path = os.path.join(dataset_output_path, "images")
        output_masks_path = os.path.join(dataset_output_path, "masks")
        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_masks_path, exist_ok=True)

        # Copy the metadata file, firts we need to create the metadata file
        metadata_output_path = os.path.join(dataset_output_path, "metadata.json")
        with open(metadata_output_path, 'w') as f:
            json.dump(metadata, f)
            
        for i, (image_path, mask_path) in enumerate(dataset_path):
            image = self.load_image(dataset_path[i][image_path])
            mask = self.load_mask(dataset_path[i][mask_path])

            image, mask = self.process_image(image, mask, mask_mapping=mask_mapping)

            image_path_output = os.path.join(output_images_path, os.path.basename(dataset_path[i][image_path]))
            mask_path_output = os.path.join(output_masks_path, os.path.basename(dataset_path[i][mask_path]))

            output_image_path = os.path.join(output_images_path, image_path_output)
            output_mask_path = os.path.join(output_masks_path, mask_path_output)
            cv2.imwrite(output_image_path, image)
            cv2.imwrite(output_mask_path, mask)