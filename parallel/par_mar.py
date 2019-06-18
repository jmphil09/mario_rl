import retro
import numpy as np
import cv2
import neat
import pickle


SHOW_GAME = False
SHOW_NN_VIEW = False
assert not (SHOW_GAME and SHOW_NN_VIEW)


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
        self.env.reset()

        #obs = self.env.reset()
        #action = self.env.action_space.sample()
        obs, _, _, _ = self.env.step(self.env.action_space.sample())

        input_x, input_y, input_colors = self.env.observation_space.shape
        input_x = int(input_x/8)
        input_y = int(input_y/8)

        #net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        print(self.genome)
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        x_position = 0
        x_position_max = 0
        imgarray = []
        done = False

        #SHOW NN
        if SHOW_NN_VIEW:
            cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:

            if SHOW_GAME:
                self.env.render()
            frame += 1

            #SHOW NN
            if SHOW_NN_VIEW:
                scaledimg = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                #scaledimg = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
                scaledimg = cv2.resize(scaledimg, (input_x, input_y))

            #print('obs.shape: {}'.format(obs.shape))
            obs = cv2.resize(obs, (input_x, input_y))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (input_x, input_y))

            #SHOW NN
            if SHOW_NN_VIEW:
                cv2.imshow('main', scaledimg)
                cv2.waitKey(1)

            #Reshape input to a 1-d list. If using keras or tf, you can
            #just use a convolution nn command
            for x in obs:
                for y in x:
                    imgarray.append(y)

            nn_output = net.activate(imgarray)

            #print(action)
            #print(len(imgarray), nn_output)

            obs, reward, done, info = self.env.step(nn_output)

            #fitness_current += reward

            imgarray.clear()

            #x_position = info['x']
            x_position = info['xscrollLo']
            if x_position > x_position_max:
                fitness_current += 1 #get a reward for moving to the right
                x_position_max = x_position

            if x_position > 100:
                fitness_current += 100000
                done = True

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(self.genome_id, fitness_current)

            #self.genome.fitness = fitness_current

        #self.env.close()
        #env.close()
        return fitness_current


def eval_genomes(genome, config):
    worker = Worker(genome, config)
    return worker.work()


config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)

p = neat.Population(config)
pe = neat.ParallelEvaluator(1, eval_genomes)

#reporting statistics
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)


