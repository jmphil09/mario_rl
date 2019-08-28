import shutil

from ConfigGenerator import ConfigGenerator
from HyperparamTuner import HyperparamTuner


if __name__ == '__main__':
    try:
        shutil.rmtree('data')
    except Exception as ex:
        print("data directory does not exist")
        print(ex)
    N = 6
    worker_start_num = 0
    worker_end_num = N
    config = ConfigGenerator()
    config_dict = config.randomize()
    config.write_all_configs(worker_start_num, worker_end_num)
    tuner = HyperparamTuner(config_dict)
    result = tuner.run_multiple_workers(worker_start_num, worker_end_num)
    score_list = []

    for n in range(0, N):
        score_list.append(tuner.print_output(n)[1])
    print(config_dict)
    print("Max Score: {}".format(max(score_list)))
    print("Average Score: {}".format(sum(score_list)/len(score_list)))
