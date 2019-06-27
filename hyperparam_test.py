from GameRunner import GameRunner


if __name__ == '__main__':
    runner = GameRunner(
        num_threads=1,
        show_game=True,
        show_nn_view=False,
        level_end_score=3186,
        convolution_weight=8,
        config_file_name='config'
    )
