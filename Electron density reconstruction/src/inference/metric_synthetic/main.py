import sys
import random

import numpy as np
import torch

sys.path.extend(["../", "../../", "../../../"])

from module.constant import data_config
from module_torch.dataset import DatasetCreate3DAdditive

from src.preprocessing import signal
from src.inference.utils import clean_inference, plot_screen, run_num, calculate_metrics, load_model, noise_inference

from src.train_electron_density_reconstruction.architecture import architecture
from src.train_electron_density_reconstruction.main import custom_augment

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

np.random.seed(seed)
random.seed(seed)

def main(exp_name, noise=False):
    data_config.set_global_key(f'{exp_name}_base')

    device = 'cuda'
    model = architecture.Model3dMultiTask()
    model = load_model(model, f'../models/tsms{exp_name}/model.pth')

    model = model.to(device)

    data = np.load(f'../../../datasets/numerical_dataset/processed/tsms{exp_name}_test_500.npz')

    i1 = data['i1']
    i2 = data['i2']
    spectrum = data['spectrum']
    angle_x = data['angle_x']
    angle_y = data['angle_y']
    angle_dist = data['angle_dist']

    energy = data_config['energy']
    spectrum = signal.resize(spectrum[:, 0, :], energy.min(), energy.max(), data_config['shapes'][0])

    dataset = DatasetCreate3DAdditive(i1, i2, spectrum, angle_x, angle_y, angle_dist, batch_size=4)

    model.eval()
    if noise:
        i1_array, i2_array, pred_array = noise_inference(model, dataset, custom_augment, NOISE_MAX=2.5 if exp_name=='18' else 2)
    else:
        i1_array, i2_array, pred_array = clean_inference(model, dataset)


    i1_pred, i2_pred, i1_array, i2_array = run_num(pred_array, i1_array, i2_array)

    calculate_metrics(i1_array, i1_pred, prefix=f'i1_{exp_name}_{"noise" if noise else "clean"}_result')
    calculate_metrics(i2_array, i2_pred, prefix=f'i2_{exp_name}_{"noise" if noise else "clean"}_result')

    plot_screen(i1_pred, i2_pred, i1_array, i2_array, d=f'plot_{exp_name}_{"noise" if noise else "clean"}')

    np.savez(
        f'save_plotsy{exp_name}_{"noise" if noise else "clean"}.npz',
        i1_pred=np.array(i1_pred),
        i2_pred=np.array(i2_pred),
        i1_array=np.array(i1),
        i2_array=np.array(i2),
        predictions=pred_array,
    )


if __name__ == '__main__':
    main(exp_name='11')
    main(exp_name='18')

    main(exp_name='11', noise=True)
    main(exp_name='18', noise=True)
