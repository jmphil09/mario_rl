from GameRunner import GameRunner


if __name__ == '__main__':
    runner = GameRunner(
        num_threads=16,
        show_game=False,
        show_nn_view=False,
        level_end_score=3186,
        convolution_weight=8,
        config_file_name='config',
        worker_start_num=0
    )

    runner.run_all_threads()
