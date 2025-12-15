import torch
from socib_shoreline_extraction.app.model.cnn_model import CNNModel
from torch import Tensor
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class DeepLabV3(CNNModel):
    """
    DeepLabV3 model for image segmentation tasks.

    This class extends the CNNModel class and provides a specific implementation
    for the DeepLabV3 architecture. It initializes the model with the
    DeepLabV3 architecture and sets the number of output classes.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        if pretrained:
            model = deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
            )
        else:
            model = deeplabv3_resnet50(weights=None)

        # Change the output layer to match the number of classes
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        super().__init__(model=model, num_classes=num_classes)

    def forward_pass(self, input_image: Tensor) -> Tensor:
        return self.model(input_image)["out"]
