#This class is to run NEATEVOLVE on config files.

import glob
import retro
import numpy as np
import cv2
import neat
import pickle

from multiprocessing import Pool
from pathlib import Path


def run(self, worker_num):

    def eval_genomes(genomes, config):

        for genome_id, genome in genomes:

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            current_max_fitness = 0
            fitness_current = 0
            frame_counter = 0

            done = False

            while not done:

                #Reshape input to a 1-d list.
                #TODO: use config values here
                imgarray = [num for row in obs for num in row]

                nn_output = net.activate(imgarray)

                obs, reward, done, info = env.step(nn_output)
                #TODO: generate output by running NN
                #reward = output from NN's

                #This reward function gives 1 point every time xscrollLo increases
                fitness_current += reward

                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    frame_counter = 0
                else:
                    frame_counter += 1

                if done or frame_counter == 250:
                    done = True

                genome.fitness = fitness_current

    #generate random config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        self.config_file_name
    )


    p = neat.Population(config)
    print('No population checkpoint found, creating new population.')

    #Show reporting statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, n=self.max_generation)

    #Save the winner
    pickle_name = Path('data/{}/complete_models/winner{}.pkl'.format(self.config_file_name, worker_num))
    pickle_dir = pickle_name.parent
    pickle_dir.mkdir(parents=True, exist_ok=True)
    with open(pickle_name, 'wb') as output:
        pickle.dump(winner, output, 1)
