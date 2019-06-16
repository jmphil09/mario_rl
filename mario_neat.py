import retro
import numpy as np
import cv2
import neat
import pickle


env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')

imgarray = []


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        obs = env.reset()
        action = env.action_space.sample()

        input_x, input_y, input_colors = env.observation_space.shape
        input_x = int(input_x/8)
        input_y = int(input_y/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        x_position = 0
        x_position_max = 0

        done = False

        while not done:

            env.render()
            frame += 1

            #print('obs.shape: {}'.format(obs.shape))
            obs = cv2.resize(obs, (input_x, input_y))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (input_x, input_y))

            #Reshape input to a 1-d list. If using keras or tf, you can
            #just use a convolution nn command
            for x in obs:
                for y in x:
                    imgarray.append(y)

            nn_output = net.activate(imgarray)

            #print(action)
            #print(len(imgarray), nn_output)

            obs, reward, done, info = env.step(nn_output)

            imgarray.clear()

            #x_position = info['x']
            x_position = info['xscrollLo']
            if x_position > x_position_max:
                fitness_current += 1 #get a reward for moving to the right
                x_position_max = x_position

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current




config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)

p = neat.Population(config)

winner = p.run(eval_genomes)
