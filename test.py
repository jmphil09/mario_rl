from HyperparamTuner import HyperparamTuner


if __name__ == '__main__':
    tuner = HyperparamTuner()

    #result = tuner.run_one_worker(14)
    result = tuner.run_multiple_workers(29, 31)
    tuner.print_output(29)
    tuner.print_output(30)
