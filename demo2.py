from GameRunner import GameRunner


if __name__ == '__main__':
    runner = GameRunner(
        num_threads=1,
        show_game=False,
        show_nn_view=True,
        level_end_score=3186,
        convolution_weight=8,
        config_file_name='config',
        worker_start_num=31,
        max_generation=100
    )

    runner.run_all_threads()
