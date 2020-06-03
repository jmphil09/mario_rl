import glob
import pickle

from pathlib import Path
from shutil import rmtree

from FitnessPlot import FitnessPlot
from GameRunner import GameRunner


DATA_FOLDER_NAME = 'data_top6'
NUMBER_TO_KEEP = 6
NUMBER_TO_RUN = 6
THREAD_START_NUM = 16
THREAD_END_NUM = 32
N = 16
M = 25


def plot_all_values(path_list=glob.glob(str(Path('hyperparam_data/*')))):
    for path in path_list:
        plot = FitnessPlot(folder_prefix=path, num_workers=8)
        plot.plot_all_workers()
        plot.plot_workers_as_average()


def generate_score_dict():
    path_list = glob.glob(str(Path('hyperparam_data/*')))
    score_dict = {}
    for path in path_list:
        score_list_for_worker = []
        files_in_folder = glob.glob(str(Path(path)) + '/*')
        for file in files_in_folder:
            if Path(file).stem != 'config' and Path(file).stem != 'config_params':
                # Calculate average of max values per worker
                with open(file, 'rb') as input_file:
                    fitness_list = pickle.load(input_file)
                    fitness_dict = {key: max(value) for (key, value) in fitness_list.items()}
                    max_key = max(fitness_dict)
                    score_list_for_worker.append(fitness_dict[max_key])
        score_dict[Path(path).stem] = sum(score_list_for_worker) / len(score_list_for_worker)
    return score_dict


def get_top_performers(n):
    score_dict = generate_score_dict()
    max_keys = sorted(score_dict, key=score_dict.get, reverse=True)[:n]
    return {key: score_dict[key] for key in max_keys}


def plot_top_performers(n, base_path='hyperparam_data/'):
    max_dict = get_top_performers(n)
    print(max_dict)
    path_list = [base_path + key for key in max_dict.keys()]
    plot_all_values(path_list)


def copy_top_config(n=1, base_path='hyperparam_data/', config_start_num=0, config_end_num=16):
    """Copy the top nth config file."""
    top_performer_dict = get_top_performers(n)
    top_performer_list = list(top_performer_dict)
    timestamp = top_performer_list[-1]
    config_file = Path(base_path + timestamp + '/config')

    with open(config_file, 'r') as orig_config:
        config_data = orig_config.read()
        for config_file_name in [Path('config' + '_' + str(worker_num)) for worker_num in range(config_start_num, config_end_num)]:
            with open(config_file_name, 'w') as fn:
                fn.write(config_data)


#clear the data folder
def clear_data_dir(dir='data'):
    try:
        rmtree(dir)
    except Exception as ex:
        print("{} directory does not exist".format(dir))
        print(ex)


def main():
    clear_data_dir(DATA_FOLDER_NAME)
    copy_top_config(n=NUMBER_TO_RUN, base_path='hyperparam_data/', config_start_num=THREAD_START_NUM, config_end_num=THREAD_END_NUM)
    runner = GameRunner(num_threads=N, show_game=True, max_generation=M, data_folder=DATA_FOLDER_NAME)
    runner.run_all_threads()


if __name__ == '__main__':
    #plot_top_performers(NUMBER_TO_KEEP)
    main()
