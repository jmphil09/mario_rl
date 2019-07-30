from ConfigGenerator import ConfigGenerator
from HyperparamTuner import HyperparamTuner


if __name__ == '__main__':
    N = 4
    worker_start_num = 0
    worker_end_num = N
    config = ConfigGenerator()
    config.write_file()
    config_dict = config.randomize()
    config.write_file()
    config.write_all_configs(worker_start_num, worker_end_num)
    #tuner = HyperparamTuner()
    tuner = HyperparamTuner(config_dict)
    #result = tuner.run_one_worker(14)
    result = tuner.run_multiple_workers(worker_start_num, worker_end_num)
    #tuner.print_output(29)
    #tuner.print_output(30)
    for n in range(0, N):
        tuner.print_output(n)

