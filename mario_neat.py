import retro
import numpy as np
import cv2
import neat
import pickle


env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')

imgarray = []

#Render the game as the NN is working
SHOW_GAME = True
#Render what the NN sees as it is working (scaled down and gray images)
SHOW_NN_VIEW = False
#TODO: see if both can be rendered without crashing
assert not (SHOW_GAME and SHOW_NN_VIEW)


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

        #SHOW NN
        if SHOW_NN_VIEW:
            cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:

            if SHOW_GAME:
                env.render()
            frame += 1

            #SHOW NN
            if SHOW_NN_VIEW:
                scaledimg = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                #scaledimg = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
                scaledimg = cv2.resize(scaledimg, (input_x, input_y))

            obs = cv2.resize(obs, (input_x, input_y))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (input_x, input_y))

            #SHOW NN
            if SHOW_NN_VIEW:
                cv2.imshow('main', scaledimg)
                cv2.waitKey(1)

            #Reshape input to a 1-d list. Note: If using keras or tf, you can
            #just use a convolution nn command
            for x in obs:
                for y in x:
                    imgarray.append(y)

            nn_output = net.activate(imgarray)

            obs, reward, done, info = env.step(nn_output)

            imgarray.clear()

            x_position = info['xscrollLo']
            if x_position > x_position_max:
                fitness_current += 1 #get a reward for moving to the right
                x_position_max = x_position

            #TODO: try this instead of looking at xscrollLo. This reward function gives
            #1 point every time xscrollLo increases, so it should work the same
            #fitness_current += reward

            #TODO: If not trying the above ^^ fitness calculation: replace 100
            #with the xscrollLo value at the end of the level (or end of the game?)
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

#show reporting statistics
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
#create a checkpoint of the NN
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

#save the winner
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
