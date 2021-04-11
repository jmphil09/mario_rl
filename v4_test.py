#from ConfigGenerator import ConfigGenerator
from GameRunner_v4 import GameRunner
#from FitnessPlot_v2 import FitnessPlot

#import glob
#from pathlib import Path

#import time

runner = GameRunner(
    num_threads=32,
    show_game=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name='good_config',
    worker_start_num=0,
    max_generation=10,
    data_folder='data_v4_test'
)

runner.run_all_threads()
