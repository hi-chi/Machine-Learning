import multiprocessing
import os
from datetime import datetime, timezone
from functools import partial

import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from module.constant import data_config
from src.data_gen.numerical_model.tsms18 import generate


data_config.set_global_key('18_base')
directories = [
    '../../../datasets/',
    '../../../datasets/numerical_dataset/',
    '../../../datasets/numerical_dataset/raw/'
]
for d in directories:
    os.makedirs(d, exist_ok=True)


save_args = {
    'directory': f'../../../datasets/numerical_dataset/raw/tsms18_train_50000/',
}

os.makedirs(save_args['directory'], exist_ok=True)

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    idx = save_args['directory'].rfind('_')
    r = range(0, int(save_args['directory'][idx + 1:-1]))
    print('cpu_count: ', multiprocessing.cpu_count())

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    start_time = datetime.now(tz=timezone.utc)

    pool.map(partial(generate, save_args=save_args, N=2_000_000, mode=0), r)

    pool.close()
    pool.join()

    end_time = datetime.now(tz=timezone.utc)
    time_elapsed = end_time - start_time

    print(f'Time: {time_elapsed}')
