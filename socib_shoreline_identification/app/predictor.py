import os
import tempfile

import cv2
import numpy as np

# Data processing imports
# from socib_shoreline_identification.app.data_processing import
from socib_shoreline_identification.app.data_processing import obtain_shoreline
from socib_shoreline_identification.app.data_processing.crop import (
    apply_masks,
    crop,
    merge_image_with_mask,
    merge_masks,
)
from socib_shoreline_identification.app.data_processing.dataset_preprocessor import (
    DatasetPreprocessor,
)
from socib_shoreline_identification.app.model.base_model import BaseModel

# Model imports
from socib_shoreline_identification.app.model.deeplab import DeepLabV3


class ShorelinePredictor:
    def __init__(
        self,
        model: str,
        model_path: str = None,
        num_classes: int = 2,
        oblique: bool = False,
    ):
        self.oblique = oblique
        self.model = self._select_model(model, num_classes)
        self.num_classes = num_classes
        self._load_model(model_path)

    def _select_model(self, model_name: str, num_classes: int) -> BaseModel:
        if model_name.lower() == "deeplabv3":
            return DeepLabV3(num_classes=num_classes)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    def _load_model(self, model_path: str):
        if model_path is not None:
            self.model.load_model(model_path)

    def _predict(
        self,
        img: np.ndarray,
        crop_coords: tuple,
        patch_size: tuple,
        stride: tuple,
        landward_pixel_gt: int,
        seaward_pixel_gt: int,
        landward_pixel_pred: int,
        seaward_pixel_pred: int,
        for_matlab: bool = False,
        mask: np.ndarray = None,
        mask_only_shoreline: bool = False,
        extract_gt_mask_coords: bool = False,
        update_pred_pixels: bool = True,
    ) -> np.ndarray:
        # 1. Extract the ROI from the input image
        roi = crop(img, crop_coords[0], crop_coords[1])
        if mask is not None:
            mask = crop(mask, crop_coords[0], crop_coords[1])

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save cropped image to temporary directory
            temp_path = os.path.join(temp_dir, "temp.jpg")
            cv2.imwrite(temp_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            # 2. Predict on the cropped image
            pred = self.model.predict_patch(
                temp_path,
                combination="avg",
                patch_size=patch_size,
                stride=stride,
                padding_mode="reflect",
            )

        # 3. Post-process and extract the shoreline
        merged_img_with_pred = merge_image_with_mask(
            roi, pred, alpha=0.7, num_classes=self.num_classes
        )
        pred_np = pred.cpu().numpy().astype(np.uint8)

        # Get shoreline from prediction
        mask_pred = obtain_shoreline.transform_mask_to_shoreline_from_img(
            pred_np, landward=landward_pixel_pred, seaward=seaward_pixel_pred
        )

        # 4. Merge with original image
        if mask is not None:
            if not mask_only_shoreline:
                mask = obtain_shoreline.transform_mask_to_shoreline_from_img(
                    mask, landward=landward_pixel_gt, seaward=seaward_pixel_gt
                )
            final_img = apply_masks(
                merged_img_with_pred,
                mask_pred,
                shoreline_pixel_predicted_mask=1,
                original_mask=mask,
                shoreline_pixel_original_mask=1,
            )
        else:
            final_img = apply_masks(
                merged_img_with_pred, mask_pred, shoreline_pixel_predicted_mask=1
            )
        img_with_pred = merge_masks(img, final_img, crop_coords[0], crop_coords[1])

        # Mask only with shoreline pixels
        full_mask_shoreline = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        full_mask_shoreline = merge_masks(
            full_mask_shoreline, mask_pred, crop_coords[0], crop_coords[1]
        )

        full_mask_pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if (
            self.num_classes == 2 and update_pred_pixels
        ):  # If we have only two classes, we add a NoData class (0) to the final mask
            aux_pred_np = np.zeros_like(pred_np)
            aux_pred_np[pred_np == 0] = 1
            aux_pred_np[pred_np == 1] = 2
            pred_np = aux_pred_np
        full_mask_pred = merge_masks(
            full_mask_pred, pred_np, crop_coords[0], crop_coords[1]
        )
        if update_pred_pixels:
            full_mask_pred[full_mask_shoreline == 1] = 255
            full_mask_pred[full_mask_pred == 1] = 75
            full_mask_pred[full_mask_pred == 2] = 150

        # 5. Obtain coords of the shoreline pixels
        shoreline_coords = np.column_stack(np.where(full_mask_shoreline == 1))
        shoreline_coords = self.format_coordinates(
            shoreline_coords, for_matlab=for_matlab
        )

        output = {
            "predicted_image": img_with_pred,
            "shoreline_mask": full_mask_shoreline,
            "predicted_mask": full_mask_pred,
            "shoreline_coords": shoreline_coords,
        }

        if extract_gt_mask_coords and mask is not None:
            original_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            original_mask = merge_masks(
                original_mask, mask, crop_coords[0], crop_coords[1]
            )
            gt_shoreline_coords = np.column_stack(np.where(original_mask == 1))
            gt_shoreline_coords = self.format_coordinates(
                gt_shoreline_coords, for_matlab=for_matlab
            )
            output["original_shoreline_coords"] = gt_shoreline_coords
        return output

    def predict(
        self,
        image_path: str,
        patch_size: tuple = (256, 512),
        stride: tuple = (128, 256),
        landward_pixel_gt: int = 0,
        seaward_pixel_gt: int = 1,
        landward_pixel_pred: int = 0,
        seaward_pixel_pred: int = 1,
    ) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = self._predict(
            img,
            ((0, 0), (img.shape[0], img.shape[1])),
            patch_size,
            stride,
            landward_pixel_gt=landward_pixel_gt,
            seaward_pixel_gt=seaward_pixel_gt,
            landward_pixel_pred=landward_pixel_pred,
            seaward_pixel_pred=seaward_pixel_pred,
            update_pred_pixels=False,
        )
        return pred

    def predict_roi(
        self, image_path: str, crop_coords: tuple, patch_size: tuple, stride: tuple
    ) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = self._predict(
            img,
            crop_coords,
            patch_size,
            stride,
            landward_pixel_pred=0,
            seaward_pixel_pred=1,
            landward_pixel_gt=0,
            seaward_pixel_gt=1,
        )
        return pred

    def predict_oblique_with_coords(
        self,
        image_path: str,
        shoreline_coords: dict,
        patch_size: tuple = (256, 512),
        stride: tuple = (128, 256),
        for_matlab: bool = False,
        extract_mask_coords: bool = False,
        landward_pixel_gt: int = 1,
        seaward_pixel_gt: int = 2,
        landward_pixel_pred: int = 0,
        seaward_pixel_pred: int = 1,
    ) -> np.ndarray:
        # print(shoreline_coords)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Extract the ROI from the input image
        height, width, _ = img.shape

        points = []
        for u, v in zip(shoreline_coords["u"], shoreline_coords["v"]):
            points.append((int(u), height - int(v)))

        ys = [p[1] for p in points]
        y_min = max(min(ys), 0)
        y_max = min(max(ys), height - 1)

        x_min = 0
        x_max = width - 1

        threshold_ratio = 0.90

        for x in range(0, x_max):
            col_pixels = img[y_min : y_max + 1, x, :]

            is_black = np.all(col_pixels == 0, axis=1)
            is_white = np.all(col_pixels == 255, axis=1)

            count_nodata = np.sum(is_black) + np.sum(is_white)
            total_pixels = col_pixels.shape[0]

            if count_nodata >= total_pixels * threshold_ratio:
                x_min += 1
            else:
                break

        for x in range(width - 1, x_min, -1):
            col_pixels = img[y_min : y_max + 1, x, :]

            is_black = np.all(col_pixels == 0, axis=1)
            is_white = np.all(col_pixels == 255, axis=1)

            count_nodata = np.sum(is_black) + np.sum(is_white)
            total_pixels = col_pixels.shape[0]

            if count_nodata >= total_pixels * threshold_ratio:
                x_max -= 1
            else:
                break

        x_max = max(x_max, x_min + 1)

        crop_coords = ((y_min, x_min), (y_max, x_max))

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(
            mask,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=1,
            thickness=1,
        )

        pred = self._predict(
            img,
            crop_coords,
            patch_size,
            stride,
            landward_pixel_gt=landward_pixel_gt,
            seaward_pixel_gt=seaward_pixel_gt,
            landward_pixel_pred=landward_pixel_pred,
            seaward_pixel_pred=seaward_pixel_pred,
            for_matlab=for_matlab,
            mask=mask,
            mask_only_shoreline=True,
        )

        shoreline_coords = np.column_stack(np.where(mask == 1))
        shoreline_coords = self.format_coordinates(
            shoreline_coords, for_matlab=for_matlab
        )

        if extract_mask_coords:
            pred["original_shoreline_coords"] = shoreline_coords

        return pred

    def predict_rectified_with_mask(
        self,
        image_path: str,
        mask_path: str,
        patch_size: tuple = (256, 256),
        stride: tuple = (128, 128),
        for_matlab: bool = False,
        extract_mask_coords: bool = False,
        landward_pixel: int = 1,
        seaward_pixel: int = 2,
    ) -> np.ndarray:
        """
        Explicit method for SCLabels dataset with rectified images and masks. Ideally is designed to extract the ROI based on the mask provided to be able to compare the results with the ground truth.
        """

        dataset_preprocessor = DatasetPreprocessor()
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mapping = {
            0: 0,  # Background → Class 0
            25: 3,  # Not classified → Class 1
            75: 1,  # Land → Class 2
            150: 2,  # Sea → Class 3
            255: 1,  # Shoreline → Class 4
        }

        new_mask = dataset_preprocessor.mask_mapping(mask, mapping)

        target_classes = [1, 2]  # Land and Sea
        hard_ignore = 3  # NotClassified

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = (
            self.extract_roi_and_bbox_strict(new_mask, target_classes, hard_ignore)
        )

        crop = ((bbox_y_min, bbox_x_min), (bbox_y_max, bbox_x_max))

        pred = self._predict(
            img,
            crop,
            patch_size,
            stride,
            landward_pixel_gt=1,
            seaward_pixel_gt=2,
            landward_pixel_pred=landward_pixel,
            seaward_pixel_pred=seaward_pixel,
            mask=new_mask,
            mask_only_shoreline=False,
            extract_gt_mask_coords=extract_mask_coords,
        )

        return pred

    def format_coordinates(
        self, shoreline_coords: np.ndarray, for_matlab: bool = False
    ) -> dict:
        # Check if the array is empty to avoid errors
        if shoreline_coords.size == 0:
            return {"u": [], "v": []}

        # The first column (index 0) from np.where is 'y' or 'v'
        v_coords = shoreline_coords[:, 0].tolist()

        # The second column (index 1) is 'x' or 'u'
        u_coords = shoreline_coords[:, 1].tolist()

        if for_matlab:
            # Convert to 1-based indexing for MATLAB compatibility
            v_coords = [v + 1 for v in v_coords]
            u_coords = [u + 1 for u in u_coords]

        return {"u": u_coords, "v": v_coords}

    def extract_roi_and_bbox_strict(self, mask, target_class_ids, hard_ignore_class_id):
        """
        Extracts a ROI by strictly excluding any row containing a 'hard ignore'
        pixel, then finds the bounding box for target classes within that ROI.

        This logic assumes that any row with a 'hard_ignore_class_id'
        (e.g., NotClassified) is completely invalid.

        Args:
            mask (np.ndarray): A 2D numpy array representing the segmentation mask.
            target_class_ids (list): Class IDs for the final bounding box
                                    (e.g., [1, 2]; Where 1 = Landwards, 2 = Seawards).
            hard_ignore_class_id (int): A class ID that invalidates any row
                                        it appears in.

        Returns:
            tuple or None: A tuple (x_min, y_min, x_max, y_max) representing the
                        bounding box coordinates, or None if no pixels with the
                        specified IDs are found.
        """
        # --- 1. Identify all fundamentally valid rows ---
        # A row is valid if it does NOT contain any 'hard_ignore_class_id' pixels.
        # np.any checks if any element in a row matches the condition.
        # The '~' inverts the result, so we get True for rows with NO hard ignores.
        is_valid_row = ~np.any(mask == hard_ignore_class_id, axis=1)

        # Find the start and end of the largest contiguous block of valid rows
        valid_row_indices = np.where(is_valid_row)[0]

        if valid_row_indices.size == 0:
            return None  # No valid rows found at all

        first_valid_row = valid_row_indices.min()
        last_valid_row = valid_row_indices.max()

        # Create a view of the mask containing only this valid block of rows
        valid_data_mask = mask[first_valid_row : last_valid_row + 1, :]
        y_trim_offset = first_valid_row

        # --- 2. Find the bounding box for target classes within the valid ROI ---
        y_indices_relative, x_indices = np.where(
            np.isin(valid_data_mask, target_class_ids)
        )

        if y_indices_relative.size == 0:
            return None  # No target pixels found in the valid region

        x_min = x_indices.min()
        x_max = x_indices.max()
        y_min_relative = y_indices_relative.min()
        y_max_relative = y_indices_relative.max()

        # --- 3. Prepare the final output ---
        bbox_x_min = x_min
        bbox_y_min = y_min_relative + y_trim_offset
        bbox_x_max = x_max
        bbox_y_max = y_max_relative + y_trim_offset

        return (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
