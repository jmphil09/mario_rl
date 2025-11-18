import glob

from multiprocessing import Pool
from pathlib import Path

from FitnessPlot import FitnessPlot
from GameRunner import GameRunner


N = 1
M = 105
DATA_FOLDER_NAME = 'data_good_candidates'


def main(worker_num):
    #plot = FitnessPlot(folder_prefix=DATA_FOLDER_NAME, worker_start_num=1, num_workers=2)
    #worker_dict = plot.create_worker_dict()
    #print(worker_dict)

    runner = GameRunner(
        num_threads=N,
        data_folder=DATA_FOLDER_NAME,
        show_game=False
    )
    runner.run_one_worker(worker_num)


def get_worker_numbers(path):
    file_list = glob.glob(str(Path(DATA_FOLDER_NAME + '/config_*')))
    return [f.split(str(Path(DATA_FOLDER_NAME + '/config_')))[1] for f in file_list]


if __name__ == '__main__':
    try:
        worker_list = get_worker_numbers(DATA_FOLDER_NAME)
        p = Pool(processes=len(worker_list))
        p.map(main, tuple(worker_list))
    except Exception as ex:
        print(ex)
