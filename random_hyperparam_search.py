import os
import pickle
import time

from pathlib import Path
from shutil import copyfile, rmtree

from ConfigGenerator import ConfigGenerator
from GameRunner import GameRunner

N = 16
M = 10


#clear the data folder
def clear_data_dir(dir='data'):
    try:
        rmtree(dir)
    except Exception as ex:
        print("{} directory does not exist".format(dir))
        print(ex)


def main():
    #Generate a random config file
    config = ConfigGenerator()
    config_params = config.randomize()

    #Copy the file N times
    config.write_all_configs(0, N)

    #Run N workers for M generations
    runner = GameRunner(num_threads=N, show_game=False, max_generation=M)
    runner.run_all_threads()

    #After all workers are finished, move the fitness list files to a data directory
    timestamp = str(time.time()).split('.')[0]
    config_path = Path('config')
    config_destination_path = Path('hyperparam_data/{}/config'.format(timestamp))
    os.makedirs(os.path.dirname(config_destination_path), exist_ok=True)
    copyfile(config_path, config_destination_path)
    with open(Path(str(config_destination_path) + '_params'), 'wb') as config_param_path:
        pickle.dump(config_params, config_param_path, 1)
    for n in range(N):
        src_path = Path('data/config_{}/worker-{}-fitness_list.pkl'.format(n, n))
        destination_path = Path('hyperparam_data/{}/worker-{}-fitness_list.pkl'.format(timestamp, n))
        copyfile(src_path, destination_path)

    clear_data_dir()


if __name__ == '__main__':
    for n in range(10):
        main()
