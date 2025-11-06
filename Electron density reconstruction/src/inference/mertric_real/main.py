import os
import sys

import numpy as np
import torch
import plotly.graph_objects as go

sys.path.extend(["../", "../../", "../../../"])

from module.constant import data_config
from module.load_and_save_data import load_realdata

from src.preprocessing.image import tsms11_cut_out_i1, tsms11_cut_out_i2, resize_opencv
from src.preprocessing.data3d import resize_3d_data

from src.inference.utils import run_num, calculate_metrics, plot_screen, load_model
from src.train_electron_density_reconstruction.architecture import architecture
from src.train_electron_density_reconstruction.fourier_distributions_transform import transform_distribution

def main(exp_name):
    data_config.set_global_key(f'{exp_name}_base')

    device = 'cuda'
    model = architecture.Model3dMultiTask()
    model = load_model(model, f'../models/tsms{exp_name}/model.pth')

    model = model.to(device)

    i1, i2 = load_realdata(f'../../../datasets/real_dataset/processed/tsms{exp_name}/')

    if exp_name == '11':
        i1 = tsms11_cut_out_i1(i1)
        i2 = tsms11_cut_out_i2(i2)

    i1 = resize_opencv(i1, (192, 64))
    i2 = resize_opencv(i2, (192, 64))

    i1 = i1 / i1.max(axis=(-1, -2), keepdims=True)
    i2 = i2 / i2.max(axis=(-1, -2), keepdims=True)

    i1 = i1[:, None]
    i2 = i2[:, None]

    predictions = []

    model.eval()

    i1_data, i2_data = torch.tensor(i1).to(device), torch.tensor(i2).to(device)

    batch_size = 4

    with torch.no_grad():
        for i in range(0, len(i1_data), batch_size):
            batch_fi1, batch_fi2 = i1_data[i:i + batch_size], i2_data[i:i + batch_size]
            outputs, _, _, _, _ = model(batch_fi1.detach(), batch_fi2.detach())

            predictions.append(outputs)

    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.detach().cpu().numpy()

    predictions = transform_distribution.inverse(predictions)
    predictions = np.clip(predictions, 0, np.inf)

    i1_pred, i2_pred, i1_array, i2_array = run_num(predictions, i1_data.detach().cpu().numpy(), i2_data.detach().cpu().numpy())

    calculate_metrics(i1_array, i1_pred, prefix=f'i1_{exp_name}_exp')
    calculate_metrics(i2_array, i2_pred, prefix=f'i2_{exp_name}_exp')

    plot_screen(i1_pred, i2_pred, i1_array, i2_array, d=f'plot_e_{exp_name}')

    os.makedirs(f'save_plotsy{exp_name}', exist_ok=True)

    np.savez(
        f'save_plotsy{exp_name}.npz',
        i1_pred=np.array(i1_pred),
        i2_pred=np.array(i2_pred),
        i1_array=np.array(i1),
        i2_array=np.array(i2),
        predictions=predictions,
    )


    for i in range(predictions.shape[0]):
        dist = resize_3d_data(predictions[i, 0], new_size=(32, 25, 25))

        x = np.linspace(data_config['energy'].min(), data_config['energy'].max(), dist.shape[0])
        y = np.linspace(data_config['rangex'][0], data_config['rangex'][1], dist.shape[1])
        z = np.linspace(data_config['rangey'][0], data_config['rangey'][1], dist.shape[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=dist.flatten(),
            isomin=np.min(dist),
            isomax=np.max(dist),
            opacity=0.1,
            surface_count=20,
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Energy'),
                yaxis=dict(title='Angle X'),
                zaxis=dict(title='Angle Y')
            ),
        )
        fig.write_html(f"save_plotsy{exp_name}/distribution_{i}.html")



if __name__ == '__main__':
    main(exp_name='11')
    main(exp_name='18')
