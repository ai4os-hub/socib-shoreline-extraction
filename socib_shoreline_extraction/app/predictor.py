import os
import tempfile

import cv2
import numpy as np

# Data processing imports
# from socib_shoreline_extraction.app.data_processing import
from socib_shoreline_extraction.app.data_processing import obtain_shoreline
from socib_shoreline_extraction.app.data_processing.crop import (
    apply_masks,
    crop,
    merge_image_with_mask,
    merge_masks,
)
from socib_shoreline_extraction.app.model.base_model import BaseModel

# Model imports
from socib_shoreline_extraction.app.model.deeplab import DeepLabV3


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
        if mask_pred is None:
            mask_pred = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)

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
        self,
        image_path: str,
        crop_coords: tuple,
        patch_size: tuple = (256, 512),
        stride: tuple = (128, 256),
        landward_pixel_gt: int = 0,
        seaward_pixel_gt: int = 1,
        landward_pixel_pred: int = 0,
        seaward_pixel_pred: int = 1,
    ) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        crop_coords = (
            (max(crop_coords[0][0], 0), max(crop_coords[0][1], 0)),
            (
                min(crop_coords[1][0], img.shape[0]),
                min(crop_coords[1][1], img.shape[1]),
            ),
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
