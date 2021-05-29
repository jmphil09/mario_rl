import glob
import gzip
import pickle

import matplotlib.pyplot as plt

from pathlib import Path


class FitnessPlot:
    def __init__(self, folder_prefix='data'):
        self.folder_prefix = folder_prefix

    def _create_fitness_list_for_checkpoint(self, checkpoint_filename):
        fitness_list = []
        with gzip.open('{}'.format(checkpoint_filename)) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            for species_id in population:
                species = species_set.get_species(species_id)
                fitness_list.append(species.fitness)
        return (generation, fitness_list)

    def create_fitness_dict(self, subfolder):
        result = {}
        file_list = glob.glob(str(Path('{}/*'.format(subfolder))))
        for file in file_list:
            key, value = self._create_fitness_list_for_checkpoint(file)
            result[key] = value
        return result

    def _plot_max_values(self):
        plt.figure(1)
        folders = glob.glob(str(Path('{}/*'.format(self.folder_prefix))))
        for subfolder in folders:
            plot_list = []
            fitness_dict = self.create_fitness_dict(subfolder)
            plot_list = sorted([(key, max([0 if x is None else x for x in value])) for key, value in fitness_dict.items()])
            plot_list = [x2 for (x1, x2) in plot_list]
            label = subfolder.split('/')[-1]
            plt.plot(plot_list, label='{}'.format(label))
        plt.ylabel('Max Fitness')
        plt.xlabel('Generation')
        plt.legend()

    def _plot_average_values(self):
        plt.figure(2)
        folders = glob.glob(str(Path('{}/*'.format(self.folder_prefix))))
        for subfolder in folders:
            plot_list = []
            fitness_dict = self.create_fitness_dict(subfolder)
            plot_list = sorted([(key, sum([0 if x is None else x for x in value]) / len(value)) for key, value in fitness_dict.items()])
            plot_list = [x2 for (x1, x2) in plot_list]
            label = subfolder.split('/')[-1]
            plt.plot(plot_list, label='{}'.format(label))
        plt.ylabel('Average Fitness')
        plt.xlabel('Generation')
        plt.legend()

    def plot_fitness_scores(self):
        self._plot_max_values()
        self._plot_average_values()
        plt.show()
