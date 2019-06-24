from multiprocessing import Pool
import glob
import retro
import numpy as np
import cv2
import neat
import pickle


class GameRunner:
    def __init__(
        self,
        num_threads=1,
        show_game=False,
        show_nn_view=False,
        level_end_score=3186,
        convolution_weight=8,
        config_file_name='config'
    ):
        self.num_threads = num_threads
        self.show_game = show_game
        self.show_nn_view = show_nn_view
        self.level_end_score = level_end_score
        self.convolution_weight = convolution_weight
        self.config_file_name = config_file_name

        #Stuff to add to docstring
        #Render the game as the NN is working
        #Render what the NN sees as it is working (scaled down and gray images)
        #TODO: see if both can be rendered without crashing
        assert not (self.show_game and self.show_nn_view)

        # Necessary on Windows, but not Mac (linux?)
        if __name__ == '__main__':
            p = Pool(processes=self.num_threads)
            p.map(self.run, tuple(range(self.num_threads)))

    def run(self, worker_num):
        env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')

        def eval_genomes(genomes, config):

            for genome_id, genome in genomes:
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

                done = False

                #SHOW NN
                if self.show_nn_view:
                    cv2.namedWindow("main", cv2.WINDOW_NORMAL)

                while not done:

                    if self.show_game:
                        env.render()
                    frame += 1

                    #SHOW NN
                    if self.show_nn_view:
                        scaledimg = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                        #scaledimg = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
                        scaledimg = cv2.resize(scaledimg, (input_x, input_y))

                    obs = cv2.resize(obs, (input_x, input_y))
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                    obs = np.reshape(obs, (input_x, input_y))

                    #SHOW NN
                    if self.show_nn_view:
                        cv2.imshow('main', scaledimg)
                        cv2.waitKey(1)

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
                        #print('genome_id: {}, fitness_current: {}'.format(genome_id, fitness_current))
                        done = True

                    genome.fitness = fitness_current

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
        p.add_reporter(
            neat.Checkpointer(
                generation_interval=5,
                time_interval_seconds=300,
                filename_prefix='data\\thread-{}-neat-checkpoint-'.format(worker_num)
                )
            )

        winner = p.run(eval_genomes)

        #Save the winner
        pickle_name = 'complete_models\\winner{}.pkl'.format(worker_num)
        with open(pickle_name, 'wb') as output:
            pickle.dump(winner, output, 1)

    #helper functions
    def _get_latest_checkpoint(self, worker):
        result = None
        file_list = glob.glob('data\\worker-{}-neat-checkpoint-*'.format(worker))
        if file_list:
            max_file_num = max([int(item.replace('data\\worker-{}-neat-checkpoint-'.format(worker), '')) for item in file_list])
            result = 'data\\worker-{}-neat-checkpoint-{}'.format(worker, max_file_num)
        return result

    def _penalize_for_dying(self, previous_info, current_info):
        #previous_lives = 2
        #Don't let mario die!
        '''current_lives = info['lives']
        if current_lives < previous_lives:
            done = True
            fitness_current = -1
        previous_lives = current_lives'''
        #reward = 100001  #This is for testing
        pass

    def _rewards_score_increase(self, previous_info, current_info):
        #previous_score = 0
        '''#Anytime mario gets a powerup (score+=1000), give an extra reward
        current_score = int(info['score'])
        #if current_score - previous_score >= 1:
        if current_score > previous_score:
            fitness_current += 25
        previous_score = current_score'''
        pass