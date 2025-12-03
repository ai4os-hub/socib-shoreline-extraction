import numpy as np
import matplotlib.pyplot as plt


def crop(img, position1, position2):
    return img[position1[0]:position2[0], position1[1]:position2[1]]

def merge_masks(img1, img2, position1, position2):
    merged = img1.copy()
    merged[position1[0]:position2[0], position1[1]:position2[1]] = img2
    return merged

def merge_image_with_mask(image, mask, alpha=0.5, num_classes = 2):
    COLOR_CLASS_0 = [165, 90, 0]
    COLOR_CLASS_1 = [0, 0, 200]
    land_index = 0 if num_classes == 2 else 1
    sea_index = 1 if num_classes == 2 else 2

    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    colored_mask[mask == land_index] = COLOR_CLASS_0 # Land
    colored_mask[mask == sea_index] = COLOR_CLASS_1 # Sea

    blended = (alpha * image + (1 - alpha) * colored_mask).astype(np.uint8)

    return blended

def apply_masks(image, predicted_mask, shoreline_pixel_predicted_mask, original_mask = None, shoreline_pixel_original_mask = None):
    # Copy original image
    overlay = image.copy()

    alpha = 1

    color = np.array([0, 255, 0], dtype=np.uint8)
    prediction_color = np.array([255, 0, 0], dtype=np.uint8)
    both_color = np.array([255, 255, 0], dtype=np.uint8)

    # Create masks for shoreline pixels
    only_predicted = predicted_mask == shoreline_pixel_predicted_mask
    if original_mask is not None and shoreline_pixel_original_mask is not None:
        only_original = original_mask == shoreline_pixel_original_mask
        both = only_predicted & only_original

    overlay[only_predicted] = (
        alpha * prediction_color + (1 - alpha) * overlay[only_predicted]
    ).astype(np.uint8)
    if original_mask is not None and shoreline_pixel_original_mask is not None:
        overlay[only_original & ~both] = (
            alpha * color + (1 - alpha) * overlay[only_original & ~both]
        ).astype(np.uint8)
        overlay[both] = (
            alpha * both_color + (1 - alpha) * overlay[both]
        ).astype(np.uint8)

    return overlay