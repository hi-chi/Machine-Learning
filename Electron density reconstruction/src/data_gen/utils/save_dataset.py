import cv2
import numpy as np


def save_year11(path, save_args, i1, i2, angle_x, angle_y, spectrum, angle_dist):

    i1 = i1[
            save_args['slice_i1_top']:save_args['slice_i1_bottom'],
            save_args['slice_i1_left']:save_args['slice_i1_right'],
            ]
    i2 = i2[
            save_args['slice_i2_top']:save_args['slice_i2_bottom'],
            save_args['slice_i2_left']:save_args['slice_i2_right'],
            ]

    i1 = cv2.resize(i1, (192, 64))
    i2 = cv2.resize(i2, (192, 64))

    np.savez_compressed(
        path,
        i1=i1.astype(np.float32),
        i2=i2.astype(np.float32),
        angle_x=np.array(angle_x).astype(np.float32),
        angle_y=np.array(angle_y).astype(np.float32),
        spectrum=spectrum.astype(np.float32),
        angle_dist=angle_dist.astype(np.float32),
    )


def save_year18(path, save_args, i1, i2, angle_x, angle_y, spectrum, angle_dist):
    i1 = cv2.resize(i1, (192, 64))
    i2 = cv2.resize(i2, (192, 64))

    np.savez_compressed(
        path,
        i1=i1.astype(np.float32),
        i2=i2.astype(np.float32),
        angle_x=np.array(angle_x).astype(np.float32),
        angle_y=np.array(angle_y).astype(np.float32),
        spectrum=spectrum.astype(np.float32),
        angle_dist=angle_dist.astype(np.float32),
    )