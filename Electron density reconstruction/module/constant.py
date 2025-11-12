import torch
import numpy as np
import threading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Singleton(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigManager(metaclass=Singleton):
    def __init__(self):
        self.configs = {
            '11_base': {
                "energy": np.linspace(10, 500, 10000),
                "rangex": (-0.15, 0.35),
                "rangey": (-0.1, 0.1),
                "shapes": (128, 64, 64),
                'name': 'tsms11',
                'BASELINE_ANGLE': 0.001,
                'ANGLE_COEFFICIENT': 30.,
            },

            '18_base': {
                "energy": np.linspace(5, 1200, 10000),
                "rangex": (-0.03, 0.04),
                "rangey": (-0.015, 0.015),
                "shapes": (128, 64, 64),
                'name': 'tsms18',
                'BASELINE_ANGLE': 0.006,
                'ANGLE_COEFFICIENT': 80.,
            },
        }
        self.global_key = None

    def set_global_key(self, key):
        if key in self.configs:
            self.global_key = key
        else:
            raise ValueError("Неверный ключ конфигурации.")

    def __getitem__(self, item):
        if self.global_key is not None:
            return self.configs[self.global_key][item]
        else:
            raise ValueError("Глобальный ключ конфигурации не установлен.")


data_config = ConfigManager()

if __name__ == '__main__':

    data_config.set_global_key('11_base')

    energy_data = data_config['energy']
    print("Данные энергии:")
    print(energy_data)
