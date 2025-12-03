import numpy as np
import cv2
from scipy.ndimage import binary_dilation

def transform_to_binary(mask, value):
    mask = np.where(mask == value, 1, 0).astype(np.uint8)
    return mask

def find_shoreline(pred, land_pixel, sea_pixel):
    shoreline = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        for j in range(1, pred.shape[1]):
            if pred[i, j] == land_pixel and pred[i, j-1] == sea_pixel:
                shoreline[i, j] = 1
            if pred[i, j] == sea_pixel and pred[i, j-1] == land_pixel:
                shoreline[i, j-1] = 1

    for j in range(pred.shape[1]):
        for i in range(1, pred.shape[0]):
            if pred[i, j] == land_pixel and pred[i-1, j] == sea_pixel:
                shoreline[i, j] = 1
            if pred[i, j] == sea_pixel and pred[i-1, j] == land_pixel:
                shoreline[i-1, j] = 1
            
    return shoreline

def find_shoreline_scipy(pred, land_pixel, sea_pixel):
    # The same logic as 'find_shoreline' but using scipy's binary_dilation for efficiency
    # 1. Create binary masks for 'land' and 'sea'
    is_land = (pred == land_pixel)
    is_sea = (pred == sea_pixel)

    # 2. Dilate the 'sea' mask
    # 'binary_dilation' expands the 'True' (sea) area by one pixel
    # in all four directions (up, down, left, right) by default.
    # This is extremely fast.
    dilated_sea = binary_dilation(is_sea)

    # 3. The shoreline is where the original LAND touches the DILATED SEA
    shoreline_mask = is_land & dilated_sea
    
    return shoreline_mask.astype(np.uint8)

def find_largest_contour(shoreline_mask):
    contours, _ = cv2.findContours(shoreline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=len)

    return largest_contour

def transform_mask_to_shoreline(pred_path, no_data=0, landward=75, seaward=150):
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    shoreline = find_shoreline(pred, landward, seaward)

    largest_contour = find_largest_contour(shoreline)

    if largest_contour is None:
        return None 

    # Create a mask for the largest contour
    largest_shoreline = np.zeros_like(shoreline)
    cv2.drawContours(largest_shoreline, [largest_contour], -1, 1)

    return largest_shoreline


def transform_mask_to_shoreline_from_img(pred, no_data=0, landward=75, seaward=150):
    shoreline = find_shoreline_scipy(pred, landward, seaward)
    # shoreline = find_shoreline(pred, landward, seaward)

    largest_contour = find_largest_contour(shoreline)

    if largest_contour is None:
        return None 

    # Create a mask for the largest contour
    largest_shoreline = np.zeros_like(shoreline)
    cv2.drawContours(largest_shoreline, [largest_contour], -1, 1, thickness=1)

    return largest_shoreline