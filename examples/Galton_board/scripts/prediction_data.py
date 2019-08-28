

"""Simulation of the experiment of droping balls in the Galton board"""


import numpy as np
import h5py


def save(file, result_data):
    """Save data set"""
    h5f = h5py.File(file, 'w')
    h5f.create_dataset('x', data=result_data)
    h5f.close()

def simulation(number_experiments, number_bins, number_balls):
    """Simulates the result of all experiments"""
    result = np.zeros(shape=(number_experiments, number_bins))

    for i in range(number_experiments):
        distribution = np.zeros(number_bins)
        for j in range(number_balls):
            random_value = np.random.sample(number_bins - 1)
            random_value = random_value[:] > 0.5
            distribution[random_value.sum()] += 1
        result[i] = distribution / number_balls
    return result

def main():
    """Main function for script data_prediction.py.

    Setting the main parameters, running simulations of the Galton board
    experiment and saving the results.
    """
    number_experiments = 1000
    number_bins = 128
    number_balls = number_bins * 300
    file = '../data/predict.h5'


    result_data = simulation(number_experiments, number_bins, number_balls)
    save(file, result_data)

if __name__ == "__main__":
    main()
