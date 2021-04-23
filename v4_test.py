#from ConfigGenerator import ConfigGenerator
from GameRunner_v4 import GameRunner
#from FitnessPlot_v2 import FitnessPlot

#import glob
#from pathlib import Path

#import time


runner = GameRunner(
    num_threads=16,
    show_game=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name='config_generated',
    worker_start_num=0,
    #max_generation=1,
    data_folder='data_v4_test'
)

runner.run_all_threads()

# ======================================================================
# Test the model DURING training
#runner.show_top_n(3)
# ======================================================================

# ======================================================================
# Test the model AFTER training

#pth = 'complete_models/winner31.pkl'
#runner.play(completed_model_path=pth)
#runner.play()

# ======================================================================

# ======================================================================
# This was used to generate the config used for this test
'''
from ConfigGenerator import ConfigGenerator

config = ConfigGenerator(
    filename='config_generated',
    fitness_criterion='max',
    fitness_threshold=100000,
    pop_size=20,
    reset_on_extinction=False,
    activation_default="sigmoid",
    activation_mutate_rate=0.0,
    activation_options="sigmoid",
    aggregation_default="sum",
    aggregation_mutate_rate=0.0,
    aggregation_options="sum",
    bias_init_mean=0.0,
    bias_init_stdev=1.0,
    bias_max_value=30.0,
    bias_min_value=-30.0,
    bias_mutate_power=0.5,
    bias_mutate_rate=0.7,
    bias_replace_rate=0.1,
    compatibility_disjoint_coefficient=1.0,
    compatibility_weight_coefficient=0.5,
    conn_add_prob=0.5,
    conn_delete_prob=0.5,
    enabled_default=True,
    enabled_mutate_rate=0.01,
    feed_forward=True,
    initial_connection="full",
    node_add_prob=0.2,
    node_delete_prob=0.2,
    num_hidden=0,
    num_inputs=840,
    num_outputs=9,
    response_init_mean=1.0,
    response_init_stdev=0.0,
    response_max_value=30.0,
    response_min_value=-30.0,
    response_mutate_power=0.0,
    response_mutate_rate=0.0,
    response_replace_rate=0.0,
    weight_init_mean=0.0,
    weight_init_stdev=1.0,
    weight_max_value=30,
    weight_min_value=-30,
    weight_mutate_power=0.5,
    weight_mutate_rate=0.8,
    weight_replace_rate=0.1,
    compatibility_threshold=3.0,
    species_fitness_func="max",
    max_stagnation=20,
    species_elitism=2,
    elitism=2,
    survival_threshold=0.2
)
#config_params = config.randomize()

#Copy the file N times
config.write_all_configs(0, 1)
'''
# ======================================================================
