from MarioRunner import MarioRunner

runner = MarioRunner(
    num_threads=1,
    show_game=False,
    show_nn_view=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name='config',
    worker_start_num=0,
    max_generation=200,
    data_folder='data',
    max_framerate=60,
    max_runtime=0,
    states=['Level1-1.state', 'Level2-1.state', 'Level3-1.state', 'Level4-1.state', 'Level5-1.state', 'Level6-1.state', 'Level7-1.state', 'Level8-1.state'],
    max_frame_wait=250
)

runner.run_all_threads()
