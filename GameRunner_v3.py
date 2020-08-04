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
        p.map(self.run, tuple(range(self.worker_start_num, self.worker_start_num + self.num_threads)))

    def run_one_worker(self, worker_num):
        self.run(worker_num)

    def one_hot_encode(self, ls):
        result = 0 #np.random.randint(0, 12)
        try:
            result = ls.index(1.0)
            print('Got a result')
            print(ls)
            time.sleep(10)
        except:
            pass
        return result

    def run(self, worker_num):
        #env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        #print(env)
        #self.config_file_name = '{}_{}'.format(self.config_file_name, worker_num)

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
                    obs = np.reshape(obs, (input_x, input_y))

                    #Reshape input to a 1-d list.
                    imgarray = [num for row in obs for num in row]

                    #There may be an issue with imgarray, the nn_output is always 0
                    nn_output = net.activate(imgarray)


                    #print('=================================================')
                    #print('HELLO')
                    #print('=================================================')
                    #print(nn_output)
                    if nn_output != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
                        print(nn_output)
                    nn_output = self.one_hot_encode(nn_output)

                    #print(env.step(nn_output))
                    obs, reward, done, info = env.step(nn_output)





                    #This reward function gives 1 point every time xscrollLo increases
                    fitness_current += reward

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

                    genome.fitness = fitness_current

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

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file_name
        )

        #Load population checkpoint if one exists
        latest_checkpoint = self._get_latest_checkpoint(worker_num)
        if latest_checkpoint:
            p = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            print('Loaded population checkpoint: {}'.format(latest_checkpoint))
        else:
            p = neat.Population(config)
            print('No population checkpoint found, creating new population.')

        #Show reporting statistics
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        #Create a checkpoint of the NN
        checkpoint_filename = Path('{}/{}/worker-{}-neat-checkpoint-'.format(self.data_folder, self.config_file_name, worker_num))
        save_dir = checkpoint_filename.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        p.add_reporter(
            neat.Checkpointer(
                generation_interval=1,
                time_interval_seconds=300,
                filename_prefix=checkpoint_filename
            )
        )

        winner = p.run(eval_genomes, n=self.max_generation)

        #Save the winner
        pickle_name = Path('{}/{}/complete_models/winner{}.pkl'.format(self.data_folder, self.config_file_name, worker_num))
        pickle_dir = pickle_name.parent
        pickle_dir.mkdir(parents=True, exist_ok=True)
        with open(pickle_name, 'wb') as output:
            pickle.dump(winner, output, 1)

    #helper functions
    def _get_latest_checkpoint(self, worker):
        result = None
        file_list = glob.glob(str(Path('{}/{}/worker-{}-neat-checkpoint-*'.format(self.data_folder, self.config_file_name, worker))))
        if file_list:
            max_file_num = max([int(item.replace(str(Path('{}/{}/worker-{}-neat-checkpoint-'.format(self.data_folder, self.config_file_name, worker))), '')) for item in file_list])
            self.generation = max_file_num
            result = str(Path('{}/{}/worker-{}-neat-checkpoint-{}'.format(self.data_folder, self.config_file_name, worker, max_file_num)))
        return result
