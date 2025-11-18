from ConfigGenerator import ConfigGenerator
from GameRunner_v2 import GameRunner
from FitnessPlot_v2 import FitnessPlot

import glob
from pathlib import Path

import time


# DONE - TODO: get the "best" config. For now just randomize

# DONE - Task1: use existing code to plot best config file results. Test the top ~5 and pick the fastest one.
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
        if worker_dict[9] > min_result:
            result[folder.split('/')[1]] = worker_dict[9]
        #plotter.plot_workers_as_average()
    return result

def get_top_times(top_results):
    result = {}
    for ts_path in top_results.keys():
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
            max_generation=10,
            data_folder='test_' + ts_path
        )
        runner.run_all_threads()
        end_time = time.time()
        print('Runtime: {0:01f} seconds'.format(end_time - start_time))
        result[ts_path] = ('{0:01f} seconds'.format(end_time - start_time), top_results[ts_path])
    return result

top_results = get_top_results(min_result=800)
#print(top_results)
times = get_top_times(top_results)

print('RUNTIMES, SCORES')
print(times)
#print('SCORES')
#print(top_results)

'''

RUNTIMES, SCORES
{
'1566017738': ('353.255976 seconds', 809.4375),
'1565252442': ('4239.686366 seconds', 808.0625),
'1565778890': ('3603.602892 seconds', 814.25),
'1565920874': ('6430.729112 seconds', 924.6875),
'1566010103': ('5932.586372 seconds', 981.75),
'1566080683': ('377.955044 seconds', 819.875),
'1566019329': ('403.268193 seconds', 854.125),
'1565775270': ('4813.206977 seconds', 1043.875),
'1565683053': ('4678.902288 seconds', 862.0625)}



'1566017738': ('353.255976 seconds', 809.4375),
'1566080683': ('377.955044 seconds', 819.875),
'1566019329': ('403.268193 seconds', 854.125),

'''
