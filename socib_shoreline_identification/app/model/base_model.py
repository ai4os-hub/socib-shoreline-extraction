import os
import torch
import torch.nn as nn
import numpy as np
import tempfile

from abc import ABC, abstractmethod
from typing import Union, Type
from torch.utils.data import Dataset
from torch import Tensor
from socib_shoreline_identification.app.data_processing.patchify import Patchify
from socib_shoreline_identification.app.data_processing.patch_reconstructor import PatchReconstructor
from socib_shoreline_identification.app.data_processing.cnn_formes import CNNFormes
from typing import Tuple

class BaseModel(ABC):
    def __init__(self, model: nn.Module, classes: int = 0, network_name: str = None) -> None:
        """
        Initializes the BaseModel object.

        Parameters:
        model (nn.Module): The model to use.
        classes (int): The number of classes in the dataset. Default is 0.
        network_name (str): The name of the network. If None, the class name of the model will be used.

        Returns:
        None
        """
        super(BaseModel, self).__init__()
        self.model = model
        self.classes = classes
        self.network_name = network_name if network_name else model.__class__.__name__

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, path: str) -> None:
        """
        Load the model from the given path.

        Parameters:
        path (str): The path to load the model.

        Returns:
        None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    @abstractmethod
    def predict(self, input_image: Tensor, raw_output = False, binary_threshold = 0.5, resize_shape = (256, 256)) -> Tensor:
        """Predicts the output for a single input image."""
        pass

    def forward_pass(self, input_image: Tensor) -> Tensor:
        """
        Forward pass for the model.

        Parameters:
        input_image (Tensor): The input image.

        Returns:
        Tensor: The output of the model.
        """
        return self.model(input_image)