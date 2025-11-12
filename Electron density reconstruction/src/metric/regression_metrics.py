import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RegressionMetrics:
    def get_regression_metrics(self, data_pairs):
        # [(name, y_true, y_pred), ...]

        metrics = {}

        for name, y_true, y_pred in data_pairs:
            y_true = y_true.reshape(y_true.shape[0], -1)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)

            metric_dict = self.calc_metric(y_true, y_pred, postfix=f' ({name})')
            metric_dict['mean' + f' ({name})'] = np.mean(y_pred)
            metric_dict['median' + f' ({name})'] = np.median(y_pred)
            metric_dict['max' + f' ({name})'] = np.max(y_pred)
            metric_dict['min' + f' ({name})'] = np.min(y_pred)
            metrics.update(metric_dict)

        metrics_df = pd.DataFrame(metrics, index=['Value']).T
        metrics_df = metrics_df.round(4)
        return metrics_df

    @staticmethod
    def calc_metric(y_true, y_pred, postfix=''):
        def cosine_distance(y_true, y_pred):
            a = y_true.reshape(-1)
            b = y_pred.reshape(-1)
            return (a @ b) / (np.sqrt(a @ a) * np.sqrt(b @ b))

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        max_diff = np.max(np.abs(y_true - y_pred))

        cos_d = cosine_distance(y_true, y_pred)

        return {
            'MSE' + postfix: mse,
            'RMSE' + postfix: rmse,
            'MAE' + postfix: mae,
            'MAX' + postfix: max_diff,
            'COSd' + postfix: cos_d,
        }
