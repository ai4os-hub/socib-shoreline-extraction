# tests/conftest.py
import os
from unittest.mock import MagicMock

import pytest

import socib_shoreline_extraction.api as api

# 1. Determine the absolute path to the real test images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECTIFIED_REAL_IMAGE_PATH = os.path.join(BASE_DIR, "data", "rectified.jpg")
OBLIQUE_REAL_IMAGE_PATH = os.path.join(BASE_DIR, "data", "oblique.jpg")


def _run_prediction(image_path, is_rectified, accept_type, extra_args=None):
    """
    Helper function to execute a prediction.
    It allows dynamic image paths and rectified flags.
    """
    # Check if the image actually exists before trying to run the model
    if not os.path.exists(image_path):
        pytest.fail(f"FATAL ERROR: Test image not found at: {image_path}")

    # Mock the input file object (API expects an object with .filename)
    input_file = MagicMock()
    input_file.filename = image_path

    # Base arguments
    args = {
        "file": input_file,
        "accept": accept_type,
        "rectified": is_rectified,  # Pass the boolean flag to the API
    }

    # Merge extra arguments if provided (e.g., for ROI testing)
    if extra_args:
        args.update(extra_args)

    print(
        f"\n[TEST] Processing {os.path.basename(image_path)} (Rectified={is_rectified}) requesting {accept_type} ..."
    )

    # Call the real API
    try:
        result = api.predict(**args)
        return result
    except Exception as e:
        pytest.fail(f"Execution of api.predict failed with error: {e}")


# Define the combinations we want to test: (Image Type, Is Rectified?, Output Format)
TEST_CASES = [
    ("rectified", True, "application/json"),
    ("rectified", True, "image/*"),
    ("oblique", False, "application/json"),
    ("oblique", False, "image/*"),
]


@pytest.fixture(params=TEST_CASES)
def real_prediction(request):
    """
    Fixture for STANDARD prediction.
    It iterates over 4 scenarios:
    1. Rectified Image -> JSON
    2. Rectified Image -> Image
    3. Oblique Image -> JSON
    4. Oblique Image -> Image
    """
    img_type, is_rectified, accept_type = request.param

    # Select the correct image path
    if img_type == "rectified":
        image_path = RECTIFIED_REAL_IMAGE_PATH
    else:
        image_path = OBLIQUE_REAL_IMAGE_PATH

    result = _run_prediction(image_path, is_rectified, accept_type)
    return result, accept_type


@pytest.fixture
def real_prediction_roi():
    """
    Fixture specifically for ROI (Region of Interest) prediction.
    We keep using the RECTIFIED image for this test to ensure coordinates match.
    """
    accept_type = "application/json"
    is_rectified = True
    image_path = RECTIFIED_REAL_IMAGE_PATH

    # Define a valid crop [x1, y1, x2, y2] within 'rectified.jpg' dimensions
    roi_args = {"boolean_crop_roi": True, "crop_roi": [100, 100, 300, 300]}

    result = _run_prediction(
        image_path, is_rectified, accept_type, extra_args=roi_args
    )
    return result, accept_type
