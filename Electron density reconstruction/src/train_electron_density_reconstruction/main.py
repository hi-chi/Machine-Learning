import os
import sys
import time

import numpy as np
import torch
from torch import GradScaler, autocast, nn
from tqdm import tqdm

sys.path.extend(["../", "../../", "../../../"])

from module.constant import data_config, device
from module.history import History
from module.load_and_save_data import load_for_3d, save_model
from module.logger import Logger
from module_torch.collect_data import Collect
from module_torch.dataset import DatasetCreate3DAdditive
from module_torch.lazy_progress_bar import LazyProgressBar, ProgressBarMetric
from module_torch.utils import optimizer_to
from module_torch.weight_init import initialize_weights
from src.metric.regression_metrics import RegressionMetrics
from src.preprocessing import signal
from src.train_electron_density_reconstruction.augmentation.augmentation import CustomAugmentation
from src.train_electron_density_reconstruction.architecture import architecture

grad_scaler = GradScaler()

custom_augment = CustomAugmentation()
exp_name = 'tsms11' # 'tsms18'
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
logger = Logger(f'{exp_name}/{exp_name}.log')

NOISE_MAX = 2 if exp_name == 'tsms11' else  2.5


def round_errors(computed_errors):
    rounded_errors = {}
    for error_key, value in computed_errors.items():
        rounded_errors[error_key] = round(value, 3)
    return rounded_errors

class Exp:
    def __init__(self):
        self.history = History()
        self.metric = RegressionMetrics()

    def load(self, data_dir):
        i1_data, i2_data, s_data, ax_data, ay_data, ad_data = load_for_3d(
            data_dir
        )

        return np.copy(i1_data), np.copy(i2_data), np.copy(s_data), np.copy(ax_data), np.copy(ay_data), np.copy(ad_data)

    @staticmethod
    def check_data(i1, i2):
        def is_contains_zero_tensor(tensor):
            if (tensor == 0).any():
                logger.warning(f'The array contains zeros: {(tensor == 0).sum()}')

        max_i1 = i1.max(axis=(1, 2, 3), keepdims=True)
        max_i2 = i2.max(axis=(1, 2, 3), keepdims=True)

        is_contains_zero_tensor(max_i1)
        is_contains_zero_tensor(max_i2)

        return i1, i2

    @staticmethod
    def np_to_dataloader(
            i1_train, i1_valid, i2_train, i2_valid, s_train, s_valid,
            ax_train, ax_valid, ay_train, ay_valid, ad_train, ad_valid,

    ):
        train_dataset = DatasetCreate3DAdditive(
            i1_data=i1_train, i2_data=i2_train, spectrum=s_train, ax=ax_train, ay=ay_train, ad=ad_train,
            batch_size=int(4 * 8 * torch.cuda.device_count()),
        )
        valid_dataset = DatasetCreate3DAdditive(
            i1_data=i1_valid, i2_data=i2_valid, spectrum=s_valid, ax=ax_valid, ay=ay_valid, ad=ad_valid,
            batch_size=int(4 * 8 * torch.cuda.device_count()),
        )
        return train_dataset, valid_dataset

    @staticmethod
    def __train(model, optimizer, dataset, epoch, pbar):
        model.train()

        for i1, i2, dist in dataset(epoch=epoch):
            ni1 = custom_augment(i1, epoch=epoch, epoch_norm=NOISE_MAX * (pbar.epochs + pbar.start_epoch))
            ni2 = custom_augment(i2, epoch=epoch, epoch_norm=NOISE_MAX * (pbar.epochs + pbar.start_epoch))

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                d_pred, i11, i12, i21, i22 = model(ni1, ni2)
                loss = 0
                loss += 1 * torch.mean((d_pred - dist) ** 2)

                loss += 0.33 * (torch.mean((i1 - i11) ** 2) + torch.mean((i2 - i12) ** 2))
                loss += 0.66 * (torch.mean((i1 - i21) ** 2) + torch.mean((i2 - i22) ** 2))

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.add_data(
                "trn", {
                    "dist": (dist, d_pred),

                    "i11": (i1, i11),
                    "i12": (i2, i12),

                    "i21": (i1, i21),
                    "i22": (i2, i22),
                },
                loss=loss,
            )
        errors = pbar.compute()['trn']
        logger.info(f'Computed errors: {round_errors(errors)}')
        return errors

    @staticmethod
    def __valid(model, dataset, pbar):
        model.eval()

        with torch.inference_mode():
            for i1, i2, dist in dataset:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    d_pred, i11, i12, i21, i22 = model(i1, i2)

                pbar.add_data(
                    "val", {
                        "dist": (dist, d_pred),

                        "i11": (i1, i11),
                        "i12": (i2, i12),

                        "i21": (i1, i21),
                        "i22": (i2, i22),
                    }
                )
        errors = pbar.compute()['val']
        logger.info(f'Computed errors: {round_errors(errors)}')
        return errors

    def train(self, model, optimizer, train_dataset, valid_dataset, n_epochs):
        time.sleep(0.1)
        error_functions = {
            "dist": {"mae": ProgressBarMetric().mean_absolute_error},
            "i11": {"mae": ProgressBarMetric().mean_absolute_error},
            "i12": {"mae": ProgressBarMetric().mean_absolute_error},

            "i21": {"mae": ProgressBarMetric().mean_absolute_error},
            "i22": {"mae": ProgressBarMetric().mean_absolute_error},
        }
        pbar = LazyProgressBar(
            start_epoch=self.history.past_size,
            epochs=n_epochs,
            error_functions=error_functions,
            states=('trn', 'val'),
        )
        for epoch in pbar:
            tr_loss = self.__train(
                model, optimizer, train_dataset, epoch, pbar
            )

            vl_loss = self.__valid(
                model, valid_dataset, pbar
            )
            self.history.train_history[epoch], self.history.valid_history[epoch] = tr_loss, vl_loss

        logger.info(f'{custom_augment.noise_stats = }')

        self.history.save_history(exp_name)

    @staticmethod
    def collect_data(model, dataset, size=None):
        collect = Collect()
        model.eval()
        custom_augment.reset_stats()

        with torch.no_grad():
            for i1, i2, dist in tqdm(dataset):
                ni1 = custom_augment(i1, epoch=1, epoch_norm=NOISE_MAX * 1)
                ni2 = custom_augment(i2, epoch=1, epoch_norm=NOISE_MAX * 1)

                with autocast(device_type='cuda', dtype=torch.bfloat16):

                    d_pred, i11, i12, i21, i22 = model(ni1, ni2)


                collect.update(
                    i1=ni1, i2=ni2,
                    i11=i11, i12=i12,
                    i21=i21, i22=i22,
                    d_true=dist, d_pred=d_pred,
                    size=size
                )
        return (
            collect.memory["i1"], collect.memory["i2"],
            collect.memory["i11"], collect.memory["i12"],
            collect.memory["i21"], collect.memory["i22"],
            collect.memory["d_true"], collect.memory["d_pred"],
        )

    @staticmethod
    def setup_model():
        model = architecture.Model3dMultiTask()
        model.apply(initialize_weights)
        print(model.state_dict().keys())

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0002,
            weight_decay=.0, amsgrad=False,
        )

        model = nn.DataParallel(model)
        model = model.to(device)
        optimizer_to(optimizer, device)
        return model, optimizer

    def run(self, train_dir, valid_dir, epoch):
        data_config.set_global_key('11_base')
        model, optimizer = self.setup_model()

        i1_train, i2_train, s_train, ax_train, ay_train, ad_train = self.load(train_dir)
        i1_valid, i2_valid, s_valid, ax_valid, ay_valid, ad_valid = self.load(valid_dir)

        i1_train, i2_train = self.check_data(i1_train, i2_train)
        i1_valid, i2_valid = self.check_data(i1_valid, i2_valid)

        s_train = signal.resize(
            s_train[:, 0, :], data_config['energy'].min(), data_config['energy'].max(),
            data_config['shapes'][0]
        )
        s_valid = signal.resize(
            s_valid[:, 0, :], data_config['energy'].min(), data_config['energy'].max(),
            data_config['shapes'][0]
        )

        train_dataset, valid_dataset = self.np_to_dataloader(
            np.copy(i1_train), np.copy(i1_valid),
            np.copy(i2_train), np.copy(i2_valid),
            np.copy(s_train), np.copy(s_valid),
            np.copy(ax_train), np.copy(ax_valid),
            np.copy(ay_train), np.copy(ay_valid),
            np.copy(ad_train), np.copy(ad_valid),
        )

        del s_train, ax_train, ay_train, ad_train, s_valid, ax_valid, ay_valid, ad_valid
        del i1_train, i1_valid, i2_train, i2_valid

        self.train(
            model=model, optimizer=optimizer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            n_epochs=epoch,
        )

        save_model(model, optimizer, f'{exp_name}/model.pth')


if __name__ == "__main__":
    exp = Exp()
    if exp_name == 'tsms11':
        data_config.set_global_key('11_base')
        train_dir = "../../../datasets/numerical_dataset/processed/tsms11_train_50000.npz"
        valid_dir = "../../../datasets/numerical_dataset/processed/tsms11_valid_1000.npz"
        epoch = 400

    elif exp_name == 'tsms18':
        data_config.set_global_key('18_base')
        train_dir = "../../../datasets/numerical_dataset/processed/tsms18_train_50000.npz"
        valid_dir = "../../../datasets/numerical_dataset/processed/tsms18_valid_1000.npz"
        epoch = 80
    else:
        raise ValueError

    exp = Exp()
    exp.run(train_dir, valid_dir, epoch)
