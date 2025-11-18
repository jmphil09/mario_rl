from GameRunner_v4 import GameRunner
from FitnessPlot_v4 import FitnessPlot


RUN_TRAINING = False
PLOT_RESULTS = True
PLAY_GAME = True


DATA_FOLDER = 'candidate3'
CONFIG_PREFIX = 'config_{}'.format(DATA_FOLDER)
RUN_TIME = 7200#3600
MAX_GENERATIONS = 10_000
NUM_THREADS = 24


runner = GameRunner(
    num_threads=NUM_THREADS,
    show_game=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name=CONFIG_PREFIX,
    worker_start_num=0,
    max_generation=MAX_GENERATIONS,
    data_folder=DATA_FOLDER,
    max_framerate=60,
    max_runtime=RUN_TIME
)

plot = FitnessPlot(num_threads=NUM_THREADS, folder_prefix=DATA_FOLDER, plot_max_score=True, max_score=3186)

#Run training
if RUN_TRAINING:
    try:
        runner.run_all_threads()
    except SystemExit as ex:
        print('Caught: {}'.format(ex))
    except Exception as e:
        print('Caught an unexpected exception: {}'.format(e))


# Plot fitness scores
if PLOT_RESULTS:
    plot.plot_fitness_scores()


# Test the model DURING training
if PLAY_GAME:
    runner.show_top_n(3)

# ======================================================================
# Test the model AFTER training
#pth = 'complete_models/winner31.pkl'
#runner.play(completed_model_path=pth)
#runner.play()
# ======================================================================
