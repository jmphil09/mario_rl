from ConfigGenerator import ConfigGenerator
from GameRunner import GameRunner

N = 8
M = 10


if __name__ == '__main__':
    #Generate a random config file
    config = ConfigGenerator()
    config.randomize()

    #Copy the file N times
    config.write_all_configs(0, N)

    #Run N workers for M generations
    runner = GameRunner(num_threads=N, show_game=True, max_generation=M)
    runner.run_all_threads()

    #After all workers are finished, move the fitness list files to
    #done/data_timestamp

    #clear the data folder

    #repeat
