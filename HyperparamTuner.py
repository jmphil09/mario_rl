import pickle
import time

from multiprocessing import Pool
from pathlib import Path

from ConfigGenerator import ConfigGenerator
from GameRunner import GameRunner


class HyperparamTuner:
    def __init__(
        self,
        config_input={
            'pop_size': 2,
            'reset_on_extinction': False,
            'activation_default': "sigmoid",
            'activation_mutate_rate': 0.0,
            'activation_options': "sigmoid",
            'aggregation_default': "sum",
            'aggregation_mutate_rate': 0.0,
            'aggregation_options': "sum",
            'bias_init_mean': 0.0,
            'bias_init_stdev': 1.0,
            'bias_max_value': 30.0,
            'bias_min_value': -30.0,
            'bias_mutate_power': 0.5,
            'bias_mutate_rate': 0.7,
            'bias_replace_rate': 0.1,
            'compatibility_disjoint_coefficient': 1.0,
            'compatibility_weight_coefficient': 0.5,
            'conn_add_prob': 0.5,
            'conn_delete_prob': 0.5,
            'enabled_default': True,
            'enabled_mutate_rate': 0.01,
            'feed_forward': True,
            'initial_connection': "full",
            'node_add_prob': 0.2,
            'node_delete_prob': 0.2,
            'num_hidden': 0,
            'response_init_mean': 1.0,
            'response_init_stdev': 0.0,
            'response_max_value': 30.0,
            'response_min_value': -30.0,
            'response_mutate_power': 0.0,
            'response_mutate_rate': 0.0,
            'response_replace_rate': 0.0,
            'weight_init_mean': 0.0,
            'weight_init_stdev': 1.0,
            'weight_max_value': 30,
            'weight_min_value': -30,
            'weight_mutate_power': 0.5,
            'weight_mutate_rate': 0.8,
            'weight_replace_rate': 0.1,
            'compatibility_threshold': 3.0,
            'species_fitness_func': "max",
            'max_stagnation': 20,
            'species_elitism': 2,
            'elitism': 2,
            'survival_threshold': 0.2
        }
    ):
        self.config_input = config_input

    def run_one_worker(self, worker_num):
        worker = worker_num

        #randomize for the first run, get from NN on following runs

        #create config file
        config = ConfigGenerator(**self.config_input)
        config.write_all_configs(config_start_num=worker, config_end_num=worker + 1)

        #run worker
        runner = GameRunner(show_game=True, max_generation=2)
        runner.run_one_worker(worker)

        #open fitness_dict pickle file
        fitness_dict = {}
        fitness_data = Path('data/config_{}/worker-{}-fitness_list.pkl'.format(worker, worker))
        with open(fitness_data, 'rb') as input_file:
            fitness_dict = pickle.load(input_file)

        #calculate score
        #For now, take the average of the max for each generation
        score = sum([max(value) for (key, value) in fitness_dict.items()]) / len(fitness_dict)

        #output = (config params, score)
        output = (list(self.config_input.values()), score)

        #save the output
        #pickle_name = Path('data/hp_output/worker_{}_{}'.format(worker_num, str(int(time.time()))))
        pickle_name = Path('data/hp_output/worker_{}'.format(worker_num))
        pickle_dir = pickle_name.parent
        pickle_dir.mkdir(parents=True, exist_ok=True)
        with open(pickle_name, 'wb') as out_file:
            pickle.dump(output, out_file, 1)

    def run_multiple_workers(self, worker_start_num, worker_end_num):
        p = Pool(processes=worker_end_num - worker_start_num)
        pool_values = tuple(range(worker_start_num, worker_end_num))
        p.map(self.run_one_worker, pool_values)

    def print_output(self, worker_num):
        data_file = Path('data/hp_output/worker_{}'.format(worker_num))
        data = pickle.load(open(data_file, 'rb'))
        print(data)
