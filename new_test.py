from ConfigGenerator import ConfigGenerator
from GameRunner_v2 import GameRunner

#TODO: get the "best" config. For now just randomize
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

runner.run_all_threads()


#Task1: use existing code to plot best config file results. Test the top ~5 and pick the fastest one.
#Task2: refactor GameRunner, consider writing server class similar to carm server
