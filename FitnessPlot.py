import pickle

import matplotlib.pyplot as plt

from pathlib import Path


class FitnessPlot:
    def __init__(self, config_file_name='config', num_workers=16, folder_prefix='data'):
        self.config_file_name = config_file_name
        self.num_workers = num_workers
        self.folder_prefix = folder_prefix

    def create_worker_dict(self):
        fitness_dict = {}
        for n in range(self.num_workers):
            worker_path = str(Path('{}/config_{}/worker-{}-fitness_list.pkl'.format(self.folder_prefix, n, n)))
            fitness_list = pickle.load(open(worker_path, 'rb'))
            fitness_dict[n] = {key: max(value) for (key, value) in fitness_list.items()}
        return fitness_dict

    def plot_all_workers(self):
        worker_dict = self.create_worker_dict()
        for n in range(self.num_workers):
            plt.plot(list(worker_dict[n].values()))
        plt.ylabel('Max Fitness')
        plt.xlabel('Generation')
        plt.show()

    def plot_workers_as_average(self):
        worker_dicts = self.create_worker_dict()
        min_dict_length = min([len(worker_dicts[n]) for n in range(self.num_workers)])
        y_values = {n: 0 for n in range(min_dict_length)}
        for n in range(self.num_workers):
            worker_dict = worker_dicts[n]
            for key in range(min_dict_length):
                y_values[key] = y_values[key] + worker_dict[key]
        y_values = {key: value/self.num_workers for key, value in y_values.items()}
        plt.plot(list(y_values.values()))
        plt.ylabel('Max Fitness')
        plt.xlabel('Generation')
        plt.show()
