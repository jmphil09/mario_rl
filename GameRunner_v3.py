import glob
import retro
import numpy as np
import cv2
import neat
import pickle

import time

from multiprocessing import Pool
from pathlib import Path

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


from pprint import saferepr


class GameRunner:
    """
    This version of the GameRunner is for running multiple levels simultaneously.
    General GameRunner class to have a NN generate a model to beat a game.

    Args:
        num_threads (int): Number of cpu threads to use
        show_game (bool): Render the frames in real-time while training
        show_nn_view (bool): Show what the Neural Network "sees" after the frame is processed
        level_end_score (int): The maximum fitness score to cause the training to end
        convolution_weight (int): Factor used to scale the image down before feeding it to the Neural Network
        config_file_name (str): The prefix to use for the config file
        worker_start_num (int): The cpu core number to start with
    """
    def __init__(
        self,
        num_threads=1,
        show_game=False,
        show_nn_view=False,
        level_end_score=3186,
        convolution_weight=8,
        config_file_name='config',
        worker_start_num=0,
        max_generation=200,
        data_folder='data'
    ):
        self.num_threads = num_threads
        self.show_game = show_game
        self.show_nn_view = show_nn_view
        self.level_end_score = level_end_score
        self.convolution_weight = convolution_weight
        self.config_file_name = config_file_name
        self.worker_start_num = worker_start_num
        self.max_generation = max_generation
        self.data_folder = data_folder

        self.fitness_scores_for_generation = []
        self.fitness_dict = {}
        self.generation = 0

    def run_all_threads(self):
        p = Pool(processes=self.num_threads)
        worker_range = range(self.worker_start_num, self.worker_start_num + self.num_threads)
        worker_levels = ['SuperMarioBros-' + str(world) + '-' + str(level) + '-v0' for world in range(1, 9) for level in range(1, 5)]
        #print('=========')
        #print(tuple(zip(worker_range, worker_levels)))
        p.map(self.run, tuple(zip(worker_range, worker_levels)))

    def run_one_worker(self, worker_num):
        self.run(worker_num)

    def one_hot_encode(self, ls):
        return ls.index(max(ls))




    '''
    Outline:
    - Load the "main" NN model - can use a config file or base model for this
    - Create a "random" set of genomes/species of the model
    Until finished:
    - Use the set of genomes/species
    - Create a tuple of the form (all_levels, all_NN_variants)
    - In parallel - run 1 episode on each worker (worker=core) over one of the tuples
    - An episode gives a reward
    - After all episodes are complete, gather results of the form dict: {NN_variant:(average of rewards on all levels)}
    - Run the "breeding" algorithm on the results to create a new set of genomes/species
    '''


    '''
    TODO:
    - Implement a function to show current progress. Do this by loading the most current model and running the highest
    score genome/species for each level.
    - Implement automated hyperparameter search (similar to ConfigGenerator class)
    '''






    def run_episode():
        pass











    def run(self, map_args):
        worker_num = map_args[0]
        level = map_args[1]
        env = gym_super_mario_bros.make(level)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)


    def eval_genomes(genomes, config):

        for genome_id, genome in genomes:
            obs = env.reset()
            #print(len(obs))
            env.action_space.sample()

            input_x, input_y, input_colors = env.observation_space.shape
            #print('Original (x,y): ({},{})'.format(input_x, input_y))
            input_x = 28#int(input_x/self.convolution_weight)
            input_y = 30#int(input_y/self.convolution_weight)
            #print('New (x,y): ({},{})'.format(input_x, input_y))

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            current_max_fitness = 0
            fitness_current = 0
            frame = 0
            frame_counter = 0

            done = False

            while not done:

                if self.show_game:
                    env.render()
                frame += 1

                obs = cv2.resize(obs, (input_x, input_y))



                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)


                #cv2.imshow('image', obs)
                #cv2.waitKey(0)


                obs = np.reshape(obs, (input_x, input_y))

                #cv2.imshow('image', obs)
                #cv2.waitKey(0)

                #Reshape input to a 1-d list.
                imgarray = [num for row in obs for num in row]

                #print(imgarray)

                #There may be an issue with imgarray, the nn_output is always 0
                nn_output = net.activate(imgarray)


                #print('=================================================')
                #print('HELLO')
                #print('=================================================')
                #print(nn_output)
                #if nn_output != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
                #    print(nn_output)
                nn_output = self.one_hot_encode(nn_output)

                #print(env.step(nn_output))
                obs, reward, done, info = env.step(nn_output)
                #print(reward)





                #This reward function gives 1 point every time xscrollLo increases
                #if reward > 0:
                fitness_current += reward
                #print('fitness_current, current_max_fitness: ({}, {})'.format(fitness_current, current_max_fitness))
                #Replace the RHS with the xscrollLo value at the end of the level
                #or end of the game
                if fitness_current > self.level_end_score:
                    fitness_current += 100000
                    done = True

                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    frame_counter = 0
                else:
                    frame_counter += 1

                if done or frame_counter == 250:
                    done = True

                #TODO: try genome.fitness = float(fitness_current)
                genome.fitness = float(fitness_current)
                #genome.fitness = float(max(fitness_current, 0))
                assert isinstance(genome.fitness, (int, float)), "Genome.fitness ({0!s}): type {1!s}, not int/float".format(saferepr(genome.fitness), type(genome.fitness))
                #print('genome.fitness: {}'.format(genome.fitness))

            self.fitness_scores_for_generation.append(fitness_current)

        fitness_list_filename = Path('{}/{}/worker-{}-fitness_list.pkl'.format(self.data_folder, self.config_file_name, worker_num))

        try:
            with open(fitness_list_filename, 'rb') as input_file:
                self.fitness_dict = pickle.load(input_file)
        except:
                self.fitness_dict = {}

        with open(fitness_list_filename, 'wb') as output:
            self.fitness_dict[self.generation] = self.fitness_scores_for_generation
            pickle.dump(self.fitness_dict, output, 1)
            self.fitness_dict = {}
            self.fitness_scores_for_generation = []
            self.generation += 1
