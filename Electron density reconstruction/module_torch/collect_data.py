import numpy as np
import torch


class Collect:
    def __init__(self):
        self.memory = {}
        self.size = 0

    def update(self, **kwargs):
        for k, v, in kwargs.items():
            if k in self.memory.keys():
                self.memory[k] = np.concatenate([self.memory[k], v.to(dtype=torch.float32).detach().cpu().numpy()], axis=0)
            else:
                self.memory[k] = v.to(dtype=torch.float32).detach().cpu().numpy()

            self.size =  self.memory[k].shape[0]

    def reset(self):
        self.memory = {}
