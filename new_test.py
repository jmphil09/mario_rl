from ConfigGenerator import ConfigGenerator
from GameRunner_v2 import GameRunner
from FitnessPlot_v2 import FitnessPlot

import glob
from pathlib import Path

import time


#TODO: get the "best" config. For now just randomize

#Task1: use existing code to plot best config file results. Test the top ~5 and pick the fastest one.
#Task2: refactor GameRunner, consider writing server class similar to carm server

'''
Using "server style" and 16 raspberry pi 4's

1) When a pi comes online, it makes a request to the server
2) The server has 32 mario levels, it "assigns" a level to the pi (more accurately, to a pi cpu core)
"assigns", means it has the pi run that level
3) The server stores the Neural Network in memory. The pi downloads the network when it is assigned a level.
4) After N runs on the level, the pi syncs its Neural Network with the main network (need to come up with strategy for weighting the networks)
'''



#config = ConfigGenerator(filename='config_test')
#config.randomize()
#config.write_all_configs(0,1)


runner = GameRunner(
    num_threads=1,
    show_game=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name='config_test',
    worker_start_num=0,
    max_generation=3,
    data_folder='data_test'
)

#runner.run_all_threads()


def get_top_results(min_result=500):
    folders = glob.glob(str(Path('hyperparam_data/*')))
    result = {}
    for folder in folders:
        #print(folder.split('/')[1])
        prefix = 'data'
        prefix = folder + '/' #'hyperparam_data/1565237431/'
        plotter = FitnessPlot(config_file_name='config', worker_start_num=0, num_workers=16, folder_prefix=prefix)
        #worker_dict = plotter.create_worker_dict()
        worker_dict = plotter.return_avg_worker_dict()
        #print(worker_dict[9])
        if worker_dict[9] > 800:
            result[folder.split('/')[1]] = worker_dict[9]
        #plotter.plot_workers_as_average()
    return result

def get_top_times(top_results):
    for ts_path in top_results.keys():
        result = {}
        prefix = str(Path('hyperparam_data/' + ts_path))
        #print(prefix)
        start_time = time.time()
        runner = GameRunner(
            num_threads=1,
            show_game=False,
            level_end_score=3186,
            convolution_weight=8,
            config_file_name=prefix + '/config',
            worker_start_num=0,
            max_generation=1,
            data_folder='test_' + ts_path
        )
        runner.run_all_threads()
        end_time = time.time()
        print('Runtime: {0:01f} seconds'.format(end_time - start_time))
        result['ts_path'] = '{0:01f}'.format(end_time - start_time)
    return result

top_results = get_top_results(min_result=800)
times = get_top_times(top_results)

print('RUNTIMES')
print(times)
