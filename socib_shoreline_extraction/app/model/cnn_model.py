import tempfile
from typing import Type

import torch
from socib_shoreline_extraction.app.data_processing.cnn_formes import CNNFormes
from socib_shoreline_extraction.app.data_processing.patch_reconstructor import (
    PatchReconstructor,
)
from socib_shoreline_extraction.app.data_processing.patchify import Patchify
from socib_shoreline_extraction.app.model.base_model import BaseModel
from torch import Tensor
from torch.utils.data import Dataset


class CNNModel(BaseModel):
    def __init__(self, model: torch.nn.Module, num_classes: int = 2):
        super().__init__(model, classes=num_classes)

    def predict(
        self,
        image_path,
        formes_class: Type[Dataset] = CNNFormes,
        raw_output=False,
        binary_threshold=0.5,
        resize_shape=(256, 256),
    ):
        """Predicts the output for a single input image."""
        self.model.to(self.device)
        self.model.eval()

        formes = formes_class(imgs_path=[image_path], resize_shape=resize_shape)
        input_image = formes[
            0
        ]  # Get the first element of the list, we only have one image

        # Add the dimension of the batch
        input_image = input_image.unsqueeze(0)

        with torch.no_grad():
            input_image = input_image.to(self.device)

            output = self.forward_pass(input_image)

        if raw_output:
            if self.classes == 1:
                return torch.sigmoid(output)
            return output

        # Compute predictions
        if self.classes > 1:
            pred = torch.argmax(output, dim=1)
        else:
            # Apply sigmoid to output to get probabilities
            prob = torch.sigmoid(output)
            pred = (prob > binary_threshold).float()

        return pred

    def predict_patch(
        self,
        image_path: str,
        patch_size: tuple = (256, 256),
        stride: tuple = (128, 128),
        formes_class: Type[Dataset] = CNNFormes,
        combination: str = "avg",
        binary_threshold=0.5,
        raw_output=False,
        padding_mode="constant",
        return_probabilities=False,
    ) -> Tensor:
        """
        Predicts the output for an image by extracting patches and reconstructing the image.

        Parameters:
        image_path (str): The path to the image file.
        patch_size (int): The size of the patches. Default: 256
        stride (int): The stride for the patches. Default: 128
        formes_class (Type[Dataset]): The class of the Form. Default: CNNFormes
        combination (str): The method to combine the patches. Options: 'avg' or 'max'. Default: 'avg'
        binary_threshold (float): The threshold to use for binary classification. Default: 0.5
        raw_output (bool): If True, return the raw output of the model. Default: False
        padding_mode (str): The padding mode to use. Default: "constant"
        return_probabilities (bool): If True, return the probabilities instead of class labels. Default: False

        Raises:
        ValueError: If the combination method is not 'avg' or 'max'.

        Returns:
        Tensor: The predicted output for the image.
        """

        # Create the patchify object
        patchify = Patchify(patch_size=patch_size, stride=stride)

        # Create a temporary directory to store the patches
        with tempfile.TemporaryDirectory() as temp_dir:
            result = patchify.extract_an_image_and_save_patches(
                image_path=image_path,
                output_image_dir=temp_dir,
                padding_mode=padding_mode,
            )

            # list of patches of the tmp directory
            input_imgs = [
                f"{temp_dir}/{patch['image_path']}" for patch in result["patches"]
            ]

            # Predict the output for each patch
            output = torch.tensor([], device=self.device)
            for input_img in input_imgs:
                raw_output_predict = self.predict(
                    input_img,
                    formes_class=formes_class,
                    raw_output=True,
                    binary_threshold=binary_threshold,
                    resize_shape=patch_size,
                )
                output = torch.cat((output, raw_output_predict), dim=0)

        # Combine the patches
        reconstruded = PatchReconstructor.combine_patches(
            output=output,
            n_classes=self.classes,
            patches=result["patches"],
            padding=result["padding"],
            patch_size=result["options"]["size"],
            stride=result["options"]["stride"],
            method=combination,
        )

        if return_probabilities:
            if self.classes == 1:
                return torch.sigmoid(reconstruded)
            return torch.softmax(reconstruded, dim=0)

        if self.classes == 1:
            if raw_output:
                return reconstruded.squeeze()
            pred = (reconstruded.squeeze() > binary_threshold).float()
            return pred

        pred = torch.argmax(reconstruded, dim=0)
        return pred
