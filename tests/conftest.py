# tests/conftest.py
import os
from unittest.mock import MagicMock

import pytest

import socib_shoreline_extraction.api as api

# 1. Determine the absolute path to the real test image
# This goes up two levels from 'tests/' to reach the project root, then into 'data/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_IMAGE_PATH = os.path.join(BASE_DIR, "data", "rectified.jpg")


@pytest.fixture(params=["application/json", "image/*"])
def real_prediction(request):
    """
    Pytest fixture that executes a REAL prediction using the image on disk.

    It runs twice automatically:
    1. Once requesting 'application/json'
    2. Once requesting 'image/png'
    """
    accept_type = request.param

    # Check if the image actually exists before trying to run the model
    if not os.path.exists(REAL_IMAGE_PATH):
        pytest.fail(f"FATAL ERROR: Test image not found at: {REAL_IMAGE_PATH}")

    # Prepare the input object simulating an uploaded file
    # The API expects an object with a .filename attribute
    input_file = MagicMock()
    input_file.filename = REAL_IMAGE_PATH

    print(
        f"\n[TEST] Loading model and processing image for accept type: {accept_type}..."
    )

    args = {"file": input_file, "accept": accept_type}

    # Call the real API (this will load the .pth model weights)
    try:
        result = api.predict(**args)
    except Exception as e:
        pytest.fail(f"Execution of api.predict failed with error: {e}")

    # Return the result and the requested type to the test function
    return result, accept_type
