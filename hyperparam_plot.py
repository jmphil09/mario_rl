import glob
import pickle

from pathlib import Path

from FitnessPlot import FitnessPlot

NUMBER_TO_KEEP = 5

def plot_all_values(path_list=glob.glob(str(Path('hyperparam_data/*')))):
    for path in path_list:
        plot = FitnessPlot(folder_prefix=path, num_workers=8)
        worker_dict = plot.create_worker_dict()

        plot.plot_all_workers()
        plot.plot_workers_as_average()


def generate_score_dict():
    path_list = glob.glob(str(Path('hyperparam_data/*')))
    score_dict = {}
    for path in path_list:
        score_list_for_worker = []
        files_in_folder = glob.glob(str(Path(path)) + '/*')
        for file in files_in_folder:
            if Path(file).stem!='config' and Path(file).stem!='config_params':
        #Calculate average of max values per worker
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
    return {key:score_dict[key] for key in max_keys}


def plot_top_performers(n, base_path='hyperparam_data/'):
    max_dict = get_top_performers(n)
    print(max_dict)
    path_list = [base_path + key for key in max_dict.keys()]
    plot_all_values(path_list)


plot_top_performers(NUMBER_TO_KEEP)
