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
from webargs import fields, validate

from socib_shoreline_identification.app.predictor import ShorelinePredictor

from . import config

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
    TODO: add more dtypes
    * int with choices
    * composed: list of strs, list of int
    """
    # WARNING: missing!=None has to go with required=False
    # fmt: off
    arg_dict = {
        "file": fields.Field(
            required=True,
            type="file",
            location="form",
            metadata={
                "description": "Input an image.\n"
                "accepted image formats: .jpg, .jpeg and .png. \n"
            },
        ), 
        "accept": fields.Str(
            required=False,
            missing="application/json",
            validate=validate.OneOf(["application/json", "image/*"]),
            metadata={
                "description": "Select the desired response format.",
            },
        ),
    }
    return arg_dict


# @_catch_error
def predict(**kwargs):
    """ """
    print("Received kwargs:", kwargs)

    image_file = kwargs.get("file")
    if image_file is None:
        raise ValueError("No image file provided in the 'file' argument.")

    image_path = image_file.filename
    print("Image path:", image_path)
    # Testing only for rectified images, 3 classes
    is_rectified = True

    path = "rectified" if is_rectified else "oblique"
    model_weight_path = os.path.abspath(
        os.path.join(os.getcwd(), f"models/{path}_best_model.pth")
    )
    print("Model weight path:", model_weight_path)

    model = "DeepLabV3"
    num_classes = 3 if is_rectified else 2

    predictor = ShorelinePredictor(model, model_weight_path, num_classes)

    landward_pixel_pred = 1 if is_rectified else 0
    seaward_pixel_pred = 2 if is_rectified else 1

    output = predictor.predict(
        image_path,
        patch_size=(256, 256),
        stride=(128, 128),
        landward_pixel_pred=landward_pixel_pred,
        seaward_pixel_pred=seaward_pixel_pred,
    )
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


# # Schema to validate the `predict()` output if accept field is "application/json"
# schema = {
#     "demo_str": fields.Str(),
#     "demo_str_choice": fields.Str(),
#     "demo_password": fields.Str(
#         metadata={
#             "format": "password",
#         },
#     ),
#     "demo_int": fields.Int(),
#     "demo_int_range": fields.Int(),
#     "demo_float": fields.Float(),
#     "demo_bool": fields.Bool(),
#     "demo_dict": fields.Dict(),
#     "demo_list_of_floats": fields.List(fields.Float()),
#     "demo_image": fields.Str(
#         description="image"  # description needed to be parsed by UI
#     ),
#     "demo_audio": fields.Str(
#         description="audio"  # description needed to be parsed by UI
#     ),
#     "demo_video": fields.Str(
#         description="video"  # description needed to be parsed by UI
#     ),
#     "labels": fields.List(fields.Str()),
#     "probabilities": fields.List(fields.Float()),
#     "accept": fields.Str(),
# }
