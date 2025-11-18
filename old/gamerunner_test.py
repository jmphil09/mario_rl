from GameRunner import GameRunner


# Constants
NUM_THREADS = 1
DATA_FOLDER = 'gamerunner_test'# + str(1) + '-2'
CONFIG_PREFIX = 'config_{}'.format(DATA_FOLDER)
RUN_TIME = 10  #3600*8#3*450  # Change first number for number of hours


# Advanced hyper parameters
CONV_WEIGHT = 16
MAX_FRAME_WAIT = 100 # was using 250 for level 3-1


#gr = GameRunner()
#gr.test_runner()


runner = GameRunner(
    num_threads=NUM_THREADS,
    show_game=False,
    show_nn_view=False,
    level_end_score=3186,
    convolution_weight=CONV_WEIGHT,
    config_file_name=CONFIG_PREFIX,
    #worker_start_num=0,
    #max_generation=MAX_GENERATIONS,
    data_folder=DATA_FOLDER,
    max_framerate=88,
    max_runtime=RUN_TIME,
    #states=STATES,
    max_frame_wait=MAX_FRAME_WAIT
)


runner.run_all_threads()
#runner.test_runner()
