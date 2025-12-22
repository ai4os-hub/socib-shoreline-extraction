# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing
the interfacing tasks. In this way you don't mix your true code with
DEEPaaS code and everything is more modular. That is, if you need to write
the predict() function in api.py, you would import your true predict function
and call it from here (with some processing / postprocessing in between
if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at an exemplar
module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

import logging
import os
from io import BytesIO
from pathlib import Path

import cv2

from socib_shoreline_extraction.app.predictor import ShorelinePredictor

from . import config, schemas

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]


def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = config.PROJECT_METADATA
        # TODO: Add dynamic metadata collection here
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


def get_predict_args():
    """
    Returns the Marshmallow Schema Class.
    DEEPaaS automatically handles the parsing when this returns a Schema.
    """
    return schemas.PredictArgsSchema().fields


def predict(**kwargs):
    """ """
    print("Received kwargs:", kwargs)

    image_file = kwargs.get("file")
    if image_file is None:
        return "No image file provided."

    is_boolean_crop_roi = kwargs.get("boolean_crop_roi", False)
    crop_roi = kwargs.get("crop_roi", None) if is_boolean_crop_roi else None
    if crop_roi is not None:
        if len(crop_roi) != 4:
            return "crop_roi must be a list of four integers: [x1, y1, x2, y2]"
        for coord in crop_roi:
            if not isinstance(coord, int):
                return "All crop_roi coordinates must be integers."

        if crop_roi[0] >= crop_roi[2] or crop_roi[1] >= crop_roi[3]:
            return "Invalid crop_roi coordinates: ensure that x2 > x1 and y2 > y1."

    image_path = image_file.filename
    print("Image path:", image_path)
    # Testing only for rectified images, 3 classes
    is_rectified = kwargs.get("rectified", True)

    path = "rectified" if is_rectified else "oblique"
    model_weight_path = os.path.abspath(
        os.path.join(os.getcwd(), f"socib-shoreline-extraction/models/{path}_best_model.pth")
    )
    print("Model weight path:", model_weight_path)

    model = "DeepLabV3"
    num_classes = 3 if is_rectified else 2

    predictor = ShorelinePredictor(model, model_weight_path, num_classes)

    landward_pixel_pred = 1 if is_rectified else 0
    seaward_pixel_pred = 2 if is_rectified else 1

    if crop_roi is None:
        output = predictor.predict(
            image_path,
            patch_size=(256, 256),
            stride=(128, 128),
            landward_pixel_pred=landward_pixel_pred,
            seaward_pixel_pred=seaward_pixel_pred,
        )
    else:
        crop_coords = ((crop_roi[1], crop_roi[0]), (crop_roi[3], crop_roi[2]))
        print("Using crop coordinates:", crop_coords)
        output = predictor.predict_roi(
            image_path,
            crop_coords=crop_coords,
            patch_size=(256, 256),
            stride=(128, 128),
            landward_pixel_pred=landward_pixel_pred,
            seaward_pixel_pred=seaward_pixel_pred,
        )
    if output is None:
        raise ValueError("Prediction failed, output is None.")
    print("Prediction completed.")
    print("Output keys:", output.keys())
    print("Output predicted_image shape:", output["predicted_image"].shape)

    if kwargs.get("accept") == "image/*":
        output["predicted_image"] = cv2.cvtColor(
            output["predicted_image"], cv2.COLOR_BGR2RGB
        )
        _, buffer = cv2.imencode(".png", output["predicted_image"])
        image_buffer = BytesIO(buffer)

        return image_buffer

    if kwargs.get("accept") == "application/json":
        return output["shoreline_coords"]

    # return message if no valid accept type is found
    return {"message": "No valid accept type found."}
