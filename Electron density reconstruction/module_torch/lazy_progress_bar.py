import numpy as np
import torch
from collections import defaultdict



class LazyProgressBar:
    def __init__(self, start_epoch, epochs, error_functions, states=("train", "valid")):
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.epoch = start_epoch

        # {'state': {'error_name': error_function}}
        self.error_functions = error_functions
        # Format for each metric, {"metric_name": ("<format>", "<color>")}

        self.states = states
        self.accumulated_errors = {state: defaultdict(lambda: {"sum": 0.0, "count": 0}) for state in self.states}
        self.loss = []

    def add_data(self, state, data, loss=None):
        if state not in self.states:
            raise ValueError(f"Invalid state '{state}'. Must be one of {self.states}.")

        for data_name, (true, pred) in data.items():
            true = true.detach()
            pred = pred.detach()

            for error_name, error_func in self.error_functions[data_name].items():
                error_value = error_func(true, pred)
                self.accumulated_errors[state][f"{state}_{error_name}({data_name})"]["sum"] += error_value
                self.accumulated_errors[state][f"{state}_{error_name}({data_name})"]["count"] += 1

        if loss:
            self.loss.append(loss.detach().cpu().numpy())

    def add_metric(self, state, metric_name, metric_value):
        if state not in self.states:
            raise ValueError(f"Invalid state '{state}'. Must be one of {self.states}.")

        # Adding metric to accumulated_errors
        self.accumulated_errors[state][metric_name]["sum"] += metric_value
        self.accumulated_errors[state][metric_name]["count"] += 1


    def compute(self):
        errors = {}
        for state in self.states:
            errors[state] = {}
            for error_key, error_data in self.accumulated_errors[state].items():
                errors[state][error_key] = error_data["sum"] / error_data["count"]

        return errors


    def __iter__(self):
        for _ in range(self.start_epoch, self.start_epoch + self.epochs):
            yield self.epoch


class ProgressBarMetric:

    @staticmethod
    def mean_absolute_error(true, pred):
        if isinstance(true, torch.Tensor):
            return torch.mean(torch.abs(true - pred)).item()
        else:
            return np.mean(np.abs(true - pred))
