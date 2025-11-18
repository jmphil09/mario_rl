from ConfigGenerator import ConfigGenerator
from GameRunner_v4 import GameRunner
from FitnessPlot_v4 import FitnessPlot


DATA_FOLDER = 'comparison_test_4'
CONFIG_PREFIX = 'config_{}'.format(DATA_FOLDER)
ITERATIONS = 6
RUN_TIME = 3600
MAX_GENERATIONS = 10_000
NUM_THREADS = 32

'''
for n in range(ITERATIONS):
    print('Working on {} out of {}'.format(n + 1, ITERATIONS))
    config = ConfigGenerator(filename='{}_{}'.format(CONFIG_PREFIX, n + 1))
    config_params = config.randomize()
    config.write_file()

    runner = GameRunner(
        num_threads=NUM_THREADS,
        show_game=False,
        level_end_score=3186,
        convolution_weight=8,
        config_file_name='{}_{}'.format(CONFIG_PREFIX, n + 1),
        worker_start_num=0,
        max_generation=MAX_GENERATIONS,
        data_folder=DATA_FOLDER,
        max_framerate=240,
        max_runtime=RUN_TIME
    )

    try:
        runner.run_all_threads()
    except SystemExit as ex:
        print('Caught: {}'.format(ex))
    except Exception as e:
        print('Caught an unexpected exception: {}'.format(e))
'''

# ======================================================================
# Plot the fitness scores for each config file
plot = FitnessPlot(num_threads=NUM_THREADS, folder_prefix=DATA_FOLDER)
plot.plot_fitness_scores()
# ======================================================================


#TODO: add a loop to show these for each config
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
