from typing import List, Optional, Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset


class CNNFormes(Dataset):
    """
    Dataset class for U-Net training and inference.

    This class loads images and their corresponding masks, applying the specified transformations.

    Attributes:
        imgs_path (List[str]): List of file paths for the input images.
        labels_path (List[str]): List of file paths for the corresponding masks.
        transform (A.Compose): Albumentations transformation pipeline.
        len (int): Number of samples in the dataset.
    """

    def __init__(
        self,
        imgs_path: List[str],
        labels_path: List[str] = None,
        transform: Optional[A.Compose] = None,
        resize_shape: Tuple[int, int] = (256, 256),
    ):
        """
        Initializes the CNNFormes dataset.

        Parameters:
            imgs_path (List[str]): List of file paths for the input images.
            labels_path (List[str]): List of file paths for the corresponding masks.
            transform (Optional[A.Compose], optional): Transformation pipeline to apply. Defaults to a standard pipeline with resizing and normalization.
            resize_shape (Tuple[int, int], optional): Desired shape to resize images and masks. Defaults to (256, 256).
        """
        super().__init__()

        self.imgs_path: List[str] = imgs_path
        self.labels_path: Optional[List[str]] = labels_path or None
        self.len: int = len(self.imgs_path)

        if transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(resize_shape[0], resize_shape[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns the image and mask at the specified index.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and the mask.
        """

        # load the image
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load the mask
        if self.labels_path is not None:
            mask = cv2.imread(self.labels_path[index], cv2.IMREAD_GRAYSCALE)
        else:
            mask = None

        # apply the transformations
        data = self.transform(image=img, mask=mask)
        image_transformed = data["image"]
        mask_transformed = data["mask"]

        if mask_transformed is None:
            return image_transformed

        # return the path of the image and the path of the label
        return image_transformed, mask_transformed

    def __len__(self):
        return self.len
