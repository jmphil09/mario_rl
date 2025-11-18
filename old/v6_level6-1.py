from GameRunner_v6 import GameRunner
from FitnessPlot_v4 import FitnessPlot


RUN_TRAINING = True
#RUN_TRAINING = False

PLOT_RESULTS = True
PLOT_RESULTS = False

PLAY_GAME = True
PLAY_GAME = False


DATA_FOLDER = 'v6_level6-1'
CONFIG_PREFIX = 'config_{}'.format(DATA_FOLDER)
RUN_TIME = 14400*4#7200#3600
MAX_GENERATIONS = 10_000
NUM_THREADS = 6

STATES = ['Level6-1.state']

runner = GameRunner(
    num_threads=NUM_THREADS,
    show_game=False,
    show_nn_view=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name=CONFIG_PREFIX,
    worker_start_num=0,
    max_generation=MAX_GENERATIONS,
    data_folder=DATA_FOLDER,
    max_framerate=60,
    max_runtime=RUN_TIME,
    states=STATES
)

plot = FitnessPlot(num_threads=NUM_THREADS, folder_prefix=DATA_FOLDER, plot_max_score=True, max_score=3074)#  max_score=3186


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


# TODO: fix
# Test the model DURING training
if PLAY_GAME:
    states = STATES
    for state in states:
        try:
            runner.show_top_n(1, show_game=True, show_nn_view=False, state=state)
        except Exception as ex:
            print(ex)

# ======================================================================
# Test the model AFTER training
#pth = 'complete_models/winner.pkl'
#runner.play(completed_model_path=pth)
#runner.play()
# ======================================================================
