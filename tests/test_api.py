# tests/test_api.py
import io
from unittest.mock import MagicMock

import pytest

import socib_shoreline_extraction.api as api

# --- METADATA TESTS ---


def test_metadata_structure():
    """
    Verifies that the metadata dictionary is correct.
    """
    meta = api.get_metadata()

    # Check type
    assert isinstance(meta, dict), "Metadata should be a dictionary"

    # Check specific values
    # assert meta["author"].lower() == "josep oliver-sanso", "Incorrect author name"
    assert meta["license"].lower() == "mit", "Incorrect license"
    assert "socib_shoreline_extraction" in meta["name"].lower().replace("-", "_")


# --- CONFIGURATION TESTS ---


def test_get_predict_args_structure():
    """
    Test to verify that get_predict_args() is defined correctly.
    This ensures high code coverage for the configuration part of api.py.
    """
    # 1. Call the function explicitly
    args = api.get_predict_args()

    # 2. Assert it returns a dictionary
    assert isinstance(args, dict), "get_predict_args should return a dict"

    # 3. Assert key arguments are present
    assert "file" in args, "Argument 'file' is missing"
    assert "rectified" in args, "Argument 'rectified' is missing"
    assert "boolean_crop_roi" in args, "Argument 'boolean_crop_roi' is missing"
    assert "crop_roi" in args, "Argument 'crop_roi' is missing"

    # 4. Check a specific default value
    assert args["accept"].load_default == "application/json", (
        "Default accept type incorrect"
    )


# --- INPUT VALIDATION TESTS (Parametrized) ---


@pytest.mark.parametrize(
    "bad_args, expected_error_msg",
    [
        # Case 1: No file provided
        ({}, "No image file provided"),
        # Case 2: Crop list has wrong length (3 items instead of 4)
        (
            {"boolean_crop_roi": True, "crop_roi": [10, 20, 30]},
            "must be a list of four integers",
        ),
        # Case 3: Crop list has wrong types (strings instead of ints)
        (
            {"boolean_crop_roi": True, "crop_roi": [10, 20, "30", 40]},
            "must be integers",
        ),
        # Case 4: Illogical coordinates (x1 > x2)
        (
            {"boolean_crop_roi": True, "crop_roi": [100, 100, 50, 200]},
            "Invalid crop_roi coordinates",
        ),
    ],
)
def test_input_validation_errors(bad_args, expected_error_msg):
    """
    Tests multiple error scenarios using parametrization.
    """
    # If the test case doesn't include 'file', we mock it to pass the first check
    if "file" not in bad_args and "No image file" not in expected_error_msg:
        bad_args["file"] = MagicMock()

    # Execute prediction
    result = api.predict(**bad_args)

    # Assert that the expected error message is in the result string
    assert expected_error_msg in result


# --- REAL INTEGRATION TESTS ---


def _validate_prediction_output(result, accept):
    """
    Helper function to assert the output format is correct.
    """
    # CASE A: If we requested an IMAGE
    if accept.startswith("image"):
        assert isinstance(result, io.BytesIO), (
            f"Expected output to be a BytesIO object, got {type(result)}"
        )
        assert result.getbuffer().nbytes > 0, "The returned image buffer is empty"

    # CASE B: If we requested JSON (coordinates)
    else:
        assert isinstance(result, dict), (
            f"Expected output to be a dict, got {type(result)}"
        )

        # Verify keys
        assert "u" in result, "Missing key 'u' in result"
        assert "v" in result, "Missing key 'v' in result"

        # Verify types
        assert isinstance(result["u"], list), "'u' should be a list"
        assert isinstance(result["v"], list), "'v' should be a list"

        # Verify consistency
        assert len(result["u"]) == len(result["v"]), "Length of 'u' and 'v' must match"

        if len(result["u"]) > 0:
            print(f" Shoreline detected with {len(result['u'])} points.")
        else:
            print(" Warning: No shoreline detected (lists are empty).")


def test_standard_prediction(real_prediction):
    """
    Integration test for the FULL IMAGE prediction path.
    Receives fixture data from conftest.py.
    """
    result, accept = real_prediction
    _validate_prediction_output(result, accept)


def test_roi_prediction(real_prediction_roi):
    """
    Integration test for the ROI (Cropped) prediction path.
    Receives fixture data from conftest.py.
    """
    result, accept = real_prediction_roi
    _validate_prediction_output(result, accept)
