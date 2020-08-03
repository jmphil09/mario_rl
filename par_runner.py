import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle

resume = False #True
restore_file = "neat-checkpoint-601"

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):

        self.env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')

        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []

        while not done:
            #cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)

            obs, reward, done, info = self.env.step(actions)

            #This reward function gives 1 point every time xscrollLo increases
            fitness_current += reward

            #Replace the RHS with the xscrollLo value at the end of the level
            #or end of the game
            if fitness_current > 3186:
                fitness_current += 100000
                done = True

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                frame_counter = 0
            else:
                frame_counter += 1

            if done or frame_counter == 250:
                done = True

            #genome.fitness = fitness_current

        print(fitness)
        return fitness

def eval_genomes(genome, config):

    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config_feedforward')

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
