import pickle

from pathlib import Path


class FitnessPlot:
    def __init__(self, config_file_name='config', num_workers=16):
        self.config_file_name = config_file_name
        self.num_workers = num_workers

    def create_worker_dict(self):
        fitness_dict = {}
        for n in range(self.num_workers):
            worker_path = str(Path('data/config_{}/worker-{}-fitness_list.pkl'.format(n, n)))
            fitness_list = pickle.load(open(worker_path, 'rb'))
            fitness_dict[n] = {key: max(value) for (key, value) in fitness_list.items()}
        return fitness_dict


plot = FitnessPlot()

print(plot.create_worker_dict())
