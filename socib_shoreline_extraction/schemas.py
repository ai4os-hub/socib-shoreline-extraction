from marshmallow import Schema, fields, validate


class PredictArgsSchema(Schema):
    class Meta:
        description = "Arguments for shoreline prediction"

    file = fields.Raw(
        required=True,
        metadata={
            "type": "file",
            "location": "form",
            "description": (
                "Input an image.\n"
                "accepted image formats: .jpg, .jpeg and .png."
            ),
        },
    )

    rectified = fields.Bool(
        load_default=True,
        metadata={
            "description": (
                "Specifies if the image is a planimetric (top-down) view with "
                "uniform ground resolution (rectified) or preserves the "
                "camera's perspective with variable resolution (oblique)."
            )
        },
    )

    boolean_crop_roi = fields.Bool(
        load_default=False,
        metadata={
            "description": (
                "Enable or disable cropping to the Region of Interest (ROI) "
                "before shoreline extraction."
            )
        },
    )

    crop_roi = fields.List(
        fields.Int(),
        load_default=[640, 480, 1000, 2000],
        validate=validate.Length(equal=4),
        metadata={
            "description": (
                "Optional Region of Interest (ROI) to crop the image. "
                "Format: [x1, y1, x2, y2]."
            )
        },
    )

    accept = fields.Str(
        load_default="application/json",
        validate=validate.OneOf(["application/json", "image/*"]),
        metadata={
            "description": "Select the desired response format.",
        },
    )
