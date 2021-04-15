import glob
import retro
import numpy as np
import cv2
import neat
import pickle
import gzip

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from multiprocessing import Pool
from pathlib import Path


class GameRunner:
    """General GameRunner class to have a NN generate a model to beat a game.

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
        self.config = None

    def run_all_threads(self):
        #p = Pool(processes=self.num_threads)
        #p.map(self.run, tuple(range(self.worker_start_num, self.worker_start_num + self.num_threads)))
        self.run(0)

    def run_one_worker(self, worker_num):
        self.run(worker_num)




    def play(self, completed_model):
        env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
        obs = env.reset()
        env.action_space.sample()

        input_x, input_y, input_colors = env.observation_space.shape
        input_x = int(input_x/self.convolution_weight)
        input_y = int(input_y/self.convolution_weight)

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file_name
        )

        net = neat.nn.recurrent.RecurrentNetwork.create(completed_model, config)

        show_game = True
        done = False

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        frame_counter = 0

        while not done:

            if show_game:
                env.render()
            frame += 1

            obs = cv2.resize(obs, (input_x, input_y))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (input_x, input_y))

            #Reshape input to a 1-d list.
            imgarray = [num for row in obs for num in row]

            nn_output = net.activate(imgarray)

            obs, reward, done, info = env.step(nn_output)

            #This reward function gives 1 point every time xscrollLo increases
            fitness_current += reward

            #Replace the RHS with the xscrollLo value at the end of the level
            #or end of the game
            if fitness_current > self.level_end_score:
                #fitness_current += 100000
                #done = True
                pass

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                frame_counter = 0
            else:
                frame_counter += 1

            if done or frame_counter == 250:
                #done = True
                pass





    def show_top_n(self, n):
        worker_num = 0


        def show_genomes(genomes, config):
            #print(genomes)
            top_genomes = []
            top_genome_id_list = []
            for genome_id, genome in genomes:
                #create genome list and sort it
                top_genome_id_list.append((genome_id, genome.fitness if genome.fitness else 0))

            top_genome_id_list = sorted(top_genome_id_list, key=lambda x: x[-1], reverse=True)[0:n]
            #top_genome_id_list.sort(key=lambda x: x[-1], reverse=True)

            #print(test)
            #print(top_genome_id_list)
            top_ids = [x1 for (x1, x2) in top_genome_id_list]
            top_genomes = [(genome_id, genome) for (genome_id, genome) in genomes if genome_id in top_ids]

            print(top_genome_id_list)
            print(top_genomes)

            #assert 1==0

            env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
            for genome_id, genome in top_genomes:
                obs = env.reset()
                env.action_space.sample()

                input_x, input_y, input_colors = env.observation_space.shape
                input_x = int(input_x/self.convolution_weight)
                input_y = int(input_y/self.convolution_weight)

                net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

                current_max_fitness = 0
                fitness_current = 0
                frame = 0
                frame_counter = 0

                show_game = True
                done = False

                while not done:

                    if show_game:
                        env.render()
                    frame += 1

                    obs = cv2.resize(obs, (input_x, input_y))
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                    obs = np.reshape(obs, (input_x, input_y))

                    #Reshape input to a 1-d list.
                    imgarray = [num for row in obs for num in row]

                    nn_output = net.activate(imgarray)

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

        #Load population checkpoint if one exists
        latest_checkpoint = self._get_latest_checkpoint(worker_num)
        if latest_checkpoint:
            pickle_file = latest_checkpoint

            #Update the generation because the implementation in neat-python has a bug
            generation = int(latest_checkpoint.split('-')[-1]) + 1
            with gzip.open(pickle_file) as f:
                contents = pickle.load(f)
                new_tuple = list(contents)
                new_tuple[0] = generation
                new_tuple = tuple(new_tuple)

            with gzip.open(pickle_file, 'w', compresslevel=5) as f:
                data = new_tuple
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            p = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            print('Loaded population checkpoint: {}'.format(latest_checkpoint))
        else:
            p = neat.Population(config)
            print('No population checkpoint found, creating new population.')


        #print(p)
        p.run(show_genomes)

        '''
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

        winner = p.run(show_genomes, n=self.max_generation)
        '''




    def eval_single_genome(self, worker_num, genome_list):
        result = set()
        env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
        #obs = env.reset()
        env.action_space.sample()
        for genome_id, genome in genome_list:

            #env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
            obs = env.reset()
            env.action_space.sample()

            input_x, input_y, input_colors = env.observation_space.shape
            input_x = int(input_x/self.convolution_weight)
            input_y = int(input_y/self.convolution_weight)

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, self.config)

            current_max_fitness = 0
            fitness_current = 0
            frame = 0
            frame_counter = 0

            done = False

            while not done:

                if self.show_game:
                    env.render()
                    #env.close()
                frame += 1

                obs = cv2.resize(obs, (input_x, input_y))
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                obs = np.reshape(obs, (input_x, input_y))

                #Reshape input to a 1-d list.
                imgarray = [num for row in obs for num in row]

                nn_output = net.activate(imgarray)

                obs, reward, done, info = env.step(nn_output)

                #This reward function gives 1 point every time xscrollLo increases
                fitness_current += reward
                #if fitness_current > 199:
                #    print(fitness_current)

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
                #env.close()

                genome.fitness = fitness_current
            result.add((genome_id, genome, fitness_current))
            #print('TODO: SAVE FITNESS SCORES HERE ({}, {}): {}'.format(genome_id, 'genome', fitness_current))
        env.close()
        return result#(genome_id, genome, fitness_current)



    def run(self, worker_num):

        def eval_genomes(genomes, config):
            split_genomes = np.array_split(genomes, self.num_threads)
            p = Pool(processes=self.num_threads)
            genome_results = p.starmap(self.eval_single_genome, [[i, (split_genomes[i])] for i in range(0, self.num_threads)])
            genome_result_list = [item for sublist in genome_results for item in sublist]
            genome_result_dict = {item[0]: (item[1], item[2]) for item in genome_result_list}

            for genome_id, genome in genomes:
                genome.fitness = genome_result_dict[genome_id][1]

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file_name
        )
        self.config = config

        #Load population checkpoint if one exists
        latest_checkpoint = self._get_latest_checkpoint(worker_num)
        if latest_checkpoint:
            pickle_file = latest_checkpoint

            #Update the generation because the implementation in neat-python has a bug
            generation = int(latest_checkpoint.split('-')[-1]) + 1
            with gzip.open(pickle_file) as f:
                contents = pickle.load(f)
                new_tuple = list(contents)
                new_tuple[0] = generation
                new_tuple = tuple(new_tuple)

            with gzip.open(pickle_file, 'w', compresslevel=5) as f:
                data = new_tuple
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

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
