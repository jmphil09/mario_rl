from GameRunner import GameRunner

N = 1
M = 100

DATA_FOLDER_NAME = 'data_top1_100gen'


def main():
    runner = GameRunner(num_threads=N, max_generation=M, data_folder=DATA_FOLDER_NAME)
    runner.run_one_worker(18)


if __name__ == '__main__':
        try:
            main()
        except Exception as ex:
            print(ex)
