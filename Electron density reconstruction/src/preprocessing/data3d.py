import numpy as np
from skimage.transform import resize

def resize_3d_data(data, new_size=(32, 25, 25)):
    if data.ndim == 3:
        resized_data = resize(
            data, new_size,
            mode='constant', preserve_range=True
        )
    else:
        resized_data = np.empty((data.shape[0], *new_size))
        for i in range(data.shape[0]):
            resized_data[i] = resize(
                data[i], new_size,
                mode='constant', preserve_range=True
            )
    return resized_data
