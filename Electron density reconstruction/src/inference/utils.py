import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from module.constant import data_config
from src.data_gen.simulation.run_num_by_pred import run_num_simulation
from src.preprocessing.image import tsms11_cut_out_i1, resize_opencv, tsms11_cut_out_i2
from src.train_electron_density_reconstruction.fourier_distributions_transform import transform_distribution


def load_model(model, file):
    checkpoint = torch.load(file)

    state_dict = checkpoint['model']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model

def cosine_distance(y_true, y_pred):
    return (y_true @ y_pred) / (np.sqrt(y_true @ y_true) * np.sqrt(y_pred @ y_pred))

def calculate_metrics(true_array, pred_array, prefix):
    mse_list = []
    mae_list = []
    cossim_list = []

    for i in range(true_array.shape[0]):
        if true_array[i].max() != 0:

            mse = mean_squared_error(true_array[i].reshape(-1), pred_array[i].reshape(-1))
            mae = mean_absolute_error(true_array[i].reshape(-1), pred_array[i].reshape(-1))
            cossim = cosine_distance(true_array[i].reshape(-1), pred_array[i].reshape(-1))

            mse_list.append(mse)
            mae_list.append(mae)
            cossim_list.append(cossim)

    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    avg_cossim = np.mean(cossim_list)
    print(f"Average MSE: {round(avg_mse, 3)}")
    print(f"Average MAE: {round(avg_mae, 3)}")
    print(f"Average Cossim: {round(avg_cossim, 3)}")

    rounded_mse = round(avg_mse, 3)
    rounded_mae = round(avg_mae, 3)
    rounded_cossim = round(avg_cossim, 3)

    if not os.path.exists('metric'):
        os.makedirs('metric')

    with open(f'metric/{prefix}.txt', 'w') as file:
        file.write(f"Average MSE: {rounded_mse}\n")
        file.write(f"Average MAE: {rounded_mae}\n")
        file.write(f"Average Cossim: {rounded_cossim}\n")


def run_num(pred_array, i1_array, i2_array):
    i1_pred, i2_pred = run_num_simulation(pred_array, N=2_000_000, shapes=(512, 256, 256))

    if data_config.global_key == '11_base':
        i1_pred = tsms11_cut_out_i1(i1_pred)
        i2_pred = tsms11_cut_out_i2(i2_pred)

    i1_pred = resize_opencv(i1_pred, (192, 64))
    i2_pred = resize_opencv(i2_pred, (192, 64))

    def normalize_array(arr):
        max_val = arr.max(axis=(-1, -2), keepdims=True)
        return np.where(max_val != 0, arr / max_val, arr)

    i1_pred = normalize_array(i1_pred)
    i2_pred = normalize_array(i2_pred)
    i1_array = normalize_array(i1_array)
    i2_array = normalize_array(i2_array)
    return i1_pred, i2_pred, i1_array, i2_array


def noise_inference(model, dataset, custom_augment, NOISE_MAX):

    dist_array = []
    pred_array = []
    i1_array = []
    i2_array = []

    with torch.no_grad():
        for i, (i1_batch, i2_batch, dist) in enumerate(dataset):
            i1_batch_ = custom_augment(i1_batch, epoch=1, epoch_norm=NOISE_MAX * 1)
            i2_batch_ = custom_augment(i2_batch, epoch=1, epoch_norm=NOISE_MAX * 1)

            pred, _, _, _, _ = model(i1_batch_, i2_batch_)
            dist_array.append(dist.detach().cpu().numpy())
            pred_array.append(pred.detach().cpu().numpy())

            i1_array.append(i1_batch.detach().cpu().numpy())
            i2_array.append(i2_batch.detach().cpu().numpy())


    pred_array = np.concatenate(pred_array)

    i1_array = np.concatenate(i1_array)
    i2_array = np.concatenate(i2_array)

    pred_array = transform_distribution.inverse(pred_array)

    pred_array = np.clip(pred_array, 0, np.inf)

    return i1_array, i2_array, pred_array

def clean_inference(model, dataset):

    dist_array = []
    pred_array = []
    i1_array = []
    i2_array = []

    with torch.no_grad():
        for i, (i1_batch, i2_batch, dist) in enumerate(dataset):

            pred, _, _, _, _ = model(i1_batch, i2_batch)
            dist_array.append(dist.detach().cpu().numpy())
            pred_array.append(pred.detach().cpu().numpy())

            i1_array.append(i1_batch.detach().cpu().numpy())
            i2_array.append(i2_batch.detach().cpu().numpy())

    pred_array = np.concatenate(pred_array)

    i1_array = np.concatenate(i1_array)
    i2_array = np.concatenate(i2_array)

    pred_array = transform_distribution.inverse(pred_array)

    pred_array = np.clip(pred_array, 0, np.inf)

    return i1_array, i2_array, pred_array

def plot_screen(i1_pred, i2_pred, i1, i2, d):
    if not os.path.exists(d):
        os.makedirs(d)

    for idx in range(min(i1_pred.shape[0], 50)):
        plt.imshow(i1_pred[idx], cmap='jet')
        plt.axis('off')
        plt.savefig(f'{d}/{idx}_i1_pred.png', bbox_inches='tight', pad_inches=0, dpi=250)
        plt.close()

        plt.imshow(i2_pred[idx], cmap='jet')
        plt.axis('off')
        plt.savefig(f'{d}/{idx}_i2_pred.png', bbox_inches='tight', pad_inches=0, dpi=250)
        plt.close()

        plt.imshow(i1[idx, 0], cmap='jet')
        plt.axis('off')
        plt.savefig(f'{d}/{idx}_i1_array.png', bbox_inches='tight', pad_inches=0, dpi=250)
        plt.close()

        plt.imshow(i2[idx, 0], cmap='jet')
        plt.axis('off')
        plt.savefig(f'{d}/{idx}_i2_array.png', bbox_inches='tight', pad_inches=0, dpi=250)
        plt.close()

