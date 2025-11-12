import cv2
import numpy as np
from scipy.ndimage import zoom

from src.preprocessing.image_constant import tsms11_dimensions_border_removal

def resize_opencv(images, new_size):
    if images.ndim == 2:
        return cv2.resize(images.astype(np.float32), new_size)
    resized_images = []

    for i in range(images.shape[0]):
        resized_img = cv2.resize(images[i].astype(np.float32), new_size)

        resized_images.append(resized_img)

    return np.array(resized_images)

def resize_scipy(images, scale_x=0.5, scale_y=0.5):
    resized_images = []
    for i in range(images.shape[0]):
        resized_img = zoom(images[i, 0].astype(np.float32), (scale_y, scale_x), order=1)[None, :, :]
        resized_images.append(resized_img)
    return np.array(resized_images, dtype=np.float16)

def tsms11_cut_out_i1(tensor):
    if tensor.ndim == 2:
        return tensor[tsms11_dimensions_border_removal["i1"]["top"]:tsms11_dimensions_border_removal["i1"]["bottom"],
                      tsms11_dimensions_border_removal["i1"]["left"]:tsms11_dimensions_border_removal["i1"]["right"]]
    elif tensor.ndim == 3:
        return tensor[:, tsms11_dimensions_border_removal["i1"]["top"]:tsms11_dimensions_border_removal["i1"]["bottom"],
                         tsms11_dimensions_border_removal["i1"]["left"]:tsms11_dimensions_border_removal["i1"]["right"]]
    elif tensor.ndim == 4:
        return tensor[:, :, tsms11_dimensions_border_removal["i1"]["top"]:tsms11_dimensions_border_removal["i1"]["bottom"],
                            tsms11_dimensions_border_removal["i1"]["left"]:tsms11_dimensions_border_removal["i1"]["right"]]

def tsms11_cut_out_i2(tensor):
    if tensor.ndim == 2:
        return tensor[tsms11_dimensions_border_removal["i2"]["top"]:tsms11_dimensions_border_removal["i2"]["bottom"],
                      tsms11_dimensions_border_removal["i2"]["left"]:tsms11_dimensions_border_removal["i2"]["right"]]
    elif tensor.ndim == 3:
        return tensor[:, tsms11_dimensions_border_removal["i2"]["top"]:tsms11_dimensions_border_removal["i2"]["bottom"],
                         tsms11_dimensions_border_removal["i2"]["left"]:tsms11_dimensions_border_removal["i2"]["right"]]
    elif tensor.ndim == 4:
        return tensor[:, :, tsms11_dimensions_border_removal["i2"]["top"]:tsms11_dimensions_border_removal["i2"]["bottom"],
                            tsms11_dimensions_border_removal["i2"]["left"]:tsms11_dimensions_border_removal["i2"]["right"]]
