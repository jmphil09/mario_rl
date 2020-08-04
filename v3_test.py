from GameRunner_v3 import GameRunner


runner = GameRunner(
    num_threads=1,
    show_game=True,
    show_nn_view=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name='config',
    worker_start_num=0,
    max_generation=200,
    data_folder='data'
)

runner.run_all_threads()
