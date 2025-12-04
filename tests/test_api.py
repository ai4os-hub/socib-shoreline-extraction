# tests/test_api.py
import io
import unittest

import socib_shoreline_extraction.api as api

# --- PART 1: METADATA TESTS (Using unittest class style) ---


class TestModelMetadata(unittest.TestCase):
    def setUp(self):
        """Run before every test method in this class."""
        self.meta = api.get_metadata()

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns a dictionary.
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns the correct specific values.
        """
        # Check the package name
        self.assertEqual(
            self.meta["name"].lower().replace("-", "_"),
            "socib_shoreline_extraction".lower().replace("-", "_"),
        )

        # Check the author
        print(f"Detected author: {self.meta['author']}")
        self.assertEqual(self.meta["author"].lower(), "Josep Oliver Sanso".lower())

        # Check the license
        self.assertEqual(
            self.meta["license"].lower(),
            "MIT".lower(),
        )


# --- PART 2: REAL PREDICTION TESTS (Using pytest function style) ---


def test_real_prediction_outputs(real_prediction):
    """
    Integration test that verifies the real model output.
    It receives the 'real_prediction' fixture from conftest.py.
    """
    result, accept = real_prediction

    # CASE A: If we requested an IMAGE
    # CHANGED: We now check if the accept string starts with "image"
    # to handle both "image/png" and "image/*"
    if accept.startswith("image"):
        # Assert it returns a BytesIO object
        assert isinstance(result, io.BytesIO), (
            f"Expected output to be a BytesIO object for images, got {type(result)}"
        )
        # Assert the image buffer is not empty
        assert result.getbuffer().nbytes > 0, "The returned image buffer is empty"

    # CASE B: If we requested JSON (coordinates)
    else:
        # 1. Verify that the result is a DICTIONARY (not a list)
        assert isinstance(result, dict), (
            f"Expected output to be a dict, got {type(result)}"
        )

        # 2. Verify that keys 'u' and 'v' exist in the result
        assert "u" in result, "Missing key 'u' in result"
        assert "v" in result, "Missing key 'v' in result"

        # 3. Verify that 'u' and 'v' values are lists
        assert isinstance(result["u"], list), "'u' value should be a list"
        assert isinstance(result["v"], list), "'v' value should be a list"

        # 4. Verify that the lists are not empty (shoreline detected)
        assert len(result["u"]) > 0, "Shoreline 'u' coordinates are empty"

        # 5. Verify data consistency: 'u' and 'v' must have the same length
        assert len(result["u"]) == len(result["v"]), "Length of 'u' and 'v' must match"

        # Optional: Print info for debugging purposes
        print(f" Shoreline detected with {len(result['u'])} points.")
