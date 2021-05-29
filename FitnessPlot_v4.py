import glob
import gzip
import pickle

import matplotlib.pyplot as plt

from pathlib import Path


class FitnessPlot:
    def __init__(self, folder_prefix='data'):
        self.config_file_name = None#config_file_name
        self.num_workers = None#num_workers
        self.folder_prefix = folder_prefix
        self.worker_start_num = 0#worker_start_num

    def _create_fitness_list_for_checkpoint(self, checkpoint_filename):
        fitness_list = []
        with gzip.open('{}'.format(checkpoint_filename)) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            for species_id in population:
                species = species_set.get_species(species_id)
                fitness_list.append(species.fitness)
        return (generation, fitness_list)

    def create_fitness_dict(self):
        result = {}
        file_list = glob.glob(str(Path('{}/*'.format(self.folder_prefix))))
        for file in file_list:
            key, value = self._create_fitness_list_for_checkpoint(file)
            result[key] = value
        return result


    def plot_max_values(self):
        plot_list = []
        fitness_dict = self.create_fitness_dict()
        plot_list = sorted([(key, max([0 if x is None else x for x in value])) for key, value in fitness_dict.items()])
        plot_list = [x2 for (x1, x2) in plot_list]
        plt.plot(plot_list, label='{}'.format(self.folder_prefix))
        plt.ylabel('Max Fitness')
        plt.xlabel('Generation')
        plt.legend()
        plt.show()

    def plot_average_values(self):
        plot_list = []
        fitness_dict = self.create_fitness_dict()
        plot_list = sorted([(key, sum([0 if x is None else x for x in value]) / len(value)) for key, value in fitness_dict.items()])
        plot_list = [x2 for (x1, x2) in plot_list]
        plt.plot(plot_list, label='{}'.format(self.folder_prefix))
        plt.ylabel('Max Fitness')
        plt.xlabel('Generation')
        plt.legend()
        plt.show()



    def create_worker_dict(self):
        fitness_dict = {}
        for n in range(self.worker_start_num, self.num_workers):
            worker_path = str(Path('{}/worker-{}-fitness_list.pkl'.format(self.folder_prefix, n)))
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

    def return_avg_worker_dict(self):
        worker_dicts = self.create_worker_dict()
        min_dict_length = min([len(worker_dicts[n]) for n in range(self.num_workers)])
        y_values = {n: 0 for n in range(min_dict_length)}
        for n in range(self.num_workers):
            worker_dict = worker_dicts[n]
            for key in range(min_dict_length):
                y_values[key] = y_values[key] + worker_dict[key]
        y_values = {key: value/self.num_workers for key, value in y_values.items()}
        return y_values
        #plt.plot(list(y_values.values()))
        #plt.ylabel('Max Fitness')
        #plt.xlabel('Generation')
        #plt.show()
