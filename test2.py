from HyperparamTuner import HyperparamTuner

if __name__ == '__main__':
    tuner = HyperparamTuner(cpu_cores=4, population=10, max_generation=3)
    tuner.tune_hyperparams()
