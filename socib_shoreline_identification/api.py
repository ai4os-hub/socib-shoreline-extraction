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

import ast
import base64
import json
import logging
import math
import mimetypes
from pathlib import Path
from random import random
import os
import shutil
import tempfile
import time

from webargs import fields, validate

from . import config, misc

from socib_shoreline_identification.app.model.deeplab import DeepLabV3

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
            validate=validate.OneOf(
                ["application/json", "application/zip", "image/*"]
            ),
            metadata={
                "description": "Select the desired response MIME type.",
            },
        ),
    }
    return arg_dict

# @_catch_error
def predict(**kwargs):
    """
    
    """
    print("Received kwargs:", kwargs)
    model = DeepLabV3(num_classes=2)
    print("Model initialized.")



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
