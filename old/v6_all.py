from GameRunner_v6 import GameRunner
from FitnessPlot_v6 import FitnessPlot


RUN_CYCLES = 100000

run_counter = 0
while run_counter < RUN_CYCLES:
    run_counter += 1

    for n in [1, 2]:  # for n in range(1, 9):    for n in [1, 4, 6]:
        RUN_TRAINING = True
        #RUN_TRAINING = False

        PLOT_RESULTS = True
        PLOT_RESULTS = False

        PLAY_GAME = True
        PLAY_GAME = False

        DATA_FOLDER = 'v6_level' + str(n) + '-1'
        CONFIG_PREFIX = 'config_{}'.format(DATA_FOLDER)
        RUN_TIME = 3*450  # Change first number for number of hours
        MAX_GENERATIONS = 100#10_000  # Change to 1 in order to use RUN_CYCLES properly
        NUM_THREADS = 48

        STATES = ['Level' + str(n) + '-1.state']

        '''
        #Temp change for level 3-1 to avoid edge case
        if STATES == ['Level3-1.state']:
            max_frame_wait = 100
        else:
            max_frame_wait = 250
        '''
        max_frame_wait = 100

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
            max_framerate=88,
            max_runtime=RUN_TIME,
            states=STATES,
            max_frame_wait=max_frame_wait
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
                    runner.show_top_n(1, show_game=True, show_nn_view=False, state=state, full_timer=False)
                except Exception as ex:
                    print(ex)

        del runner
        del plot

        # ======================================================================
        # Test the model AFTER training
        #pth = 'complete_models/winner.pkl'
        #runner.play(completed_model_path=pth)
        #runner.play()
        # ======================================================================
