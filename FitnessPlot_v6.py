import glob
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from pathlib import Path

import re
import os


class FitnessPlot:
    def __init__(self, num_threads=1, folder_prefix='data', plot_max_score=False, max_score=3186):
        self.num_threads = num_threads
        self.folder_prefix = folder_prefix
        self.plot_max_score = plot_max_score
        self.max_score = max_score

    def _create_fitness_list_for_checkpoint(self, checkpoint_filename):
        ignore_files = ['complete_models', 'fitness_dict.pkl']
        bare_file = checkpoint_filename.split('/')[-1]
        if bare_file in ignore_files:
            print('Preventing error on: {}'.format(checkpoint_filename))
            return (None, None)
        else:
            try:
                fitness_list = []
                with gzip.open('{}'.format(checkpoint_filename)) as f:
                    generation, config, population, species_set, rndstate = pickle.load(f)
                    for species_id in population:
                        species = species_set.get_species(species_id)
                        fitness_list.append(species.fitness)
                return (generation, fitness_list)
            except Exception as ex:
                print(ex)
                print('Error occurred on: {}'.format(checkpoint_filename))
                return (None, None)

    def _create_fitness_list_for_checkpoint_par(self, core_num, file_list):
        result = {}
        for file in file_list:
            key, value = self._create_fitness_list_for_checkpoint(file)
            if key:
                result[key] = value
            else:
                pass
        return result

    def create_fitness_dict_par(self, subfolder):
        result = {}
        file_list = glob.glob(str(Path('{}/*'.format(subfolder))))

        split_file_list = np.array_split(file_list, self.num_threads)
        p = Pool(processes=self.num_threads)
        result = p.starmap(self._create_fitness_list_for_checkpoint_par, [[i, (split_file_list[i])] for i in range(0, self.num_threads)])
        result = {key: value for d in result for key, value in d.items()}
        return result

    def create_fitness_dict(self, subfolder):
        result = {}
        file_list = glob.glob(str(Path('{}/*'.format(subfolder))))
        for file in file_list:
            key, value = self._create_fitness_list_for_checkpoint(file)
            result[key] = value
        return result

    def create_fitness_dict_with_cache(self, subfolder):
        # check for existing pickled fitness list
        existing_fitness_dict = self._load_fitness_dict_from_pickle(subfolder)
        # check for new checkpoint files (see if there are 2 or more?)
        # update existing pickled fitness list by adding new checkpoint data
        new_fitness_dict = self.create_fitness_dict_par(subfolder)
        new_fitness_dict.update(existing_fitness_dict)
        # Save successfully before deleting files
        try:
            delete_files = True
            self._save_fitness_dict_as_pickle(new_fitness_dict, subfolder)
        except Exception as ex:
            delete_files = False
            print(ex)
            print('Error saving fitness dict, will not delete files.')
        if delete_files:
            # delete added checkpoint files
            self._delete_checkpoint_files(subfolder)
        # return updated pickled fitness list
        return new_fitness_dict


    def _save_fitness_dict_as_pickle(self, fitness_dict, subfolder):
        pickle_name = Path('{}/fitness_dict.pkl'.format(subfolder))
        with gzip.open(pickle_name, 'w', compresslevel=5) as f:
            data = fitness_dict
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_fitness_dict_from_pickle(self, subfolder):
        result = {}
        try:
            pickle_name = Path('{}/fitness_dict.pkl'.format(subfolder))
            with gzip.open(pickle_name) as output:
                result = pickle.load(output)
        except Exception as ex:
            print(ex)
        return result

    def _delete_checkpoint_files(self, subfolder):
        folders = glob.glob(str(Path('{}/*'.format(subfolder))))
        files = [file for file in folders if 'neat-checkpoint-' in file]
        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [ atoi(c) for c in re.split('(\d+)',text) ]
        files.sort(key=natural_keys)
        files_to_keep = files[-1]
        files_to_delete = [f for f in files if f not in [files_to_keep]]
        print('Files to delete')
        print(files_to_delete)
        print('Files to keep')
        print(files_to_keep)
        for file in files_to_delete:
            print('Deleting: {}'.format(file))
            os.remove(file)

    def _plot_max_and_avg(self):
        '''
        This function plots both the max and average fitness scores.
        '''
        folders = glob.glob(str(Path('{}/*'.format(self.folder_prefix))))
        max_gen = 1
        for subfolder in folders:
            plot_list = []
            fitness_dict = self.create_fitness_dict_with_cache(subfolder)
            max_gen = max(max_gen, len(fitness_dict))
            plt.figure(1)
            plot_list = sorted([(key, max([0 if x is None else x for x in value])) for key, value in fitness_dict.items()])
            plot_list = [x2 for (x1, x2) in plot_list]
            label = subfolder.split('/')[-1]
            plt.plot(plot_list, label='{}'.format(label))

            plt.figure(2)
            plot_list = sorted([(key, sum([0 if x is None else x for x in value]) / len(value)) for key, value in fitness_dict.items()])
            plot_list = [x2 for (x1, x2) in plot_list]
            label = subfolder.split('/')[-1]
            plt.plot(plot_list, label='{}'.format(label))

        end_plot_list =[self.max_score] * max_gen
        plt.figure(1)
        if self.plot_max_score:
            plt.plot(end_plot_list, label='{}'.format('Goal: 1-1'))
        plt.ylabel('Max Fitness')
        plt.xlabel('Generation')
        plt.figure(2)
        if self.plot_max_score:
            plt.plot(end_plot_list, label='{}'.format('Goal'))
        plt.ylabel('Average Fitness')
        plt.xlabel('Generation')
        plt.legend()

    def clean_up_data(self):
        self._plot_max_and_avg()

    def plot_fitness_scores(self):
        self._plot_max_and_avg()
        plt.show()
