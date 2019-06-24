from multiprocessing import Pool
import glob
import retro
import numpy as np
import cv2
import neat
import pickle


NUMBER_OF_THREADS = 16

def run(values):
    #Global variables
    #Render the game as the NN is working
    SHOW_GAME = False
    #Render what the NN sees as it is working (scaled down and gray images)
    SHOW_NN_VIEW = False
    #TODO: see if both can be rendered without crashing
    assert not (SHOW_GAME and SHOW_NN_VIEW)

    env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')

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
            frame_counter = 0
            previous_lives = 2
            previous_score = 0

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

                #Reshape input to a 1-d list.
                imgarray = [num for row in obs for num in row]

                nn_output = net.activate(imgarray)

                obs, reward, done, info = env.step(nn_output)

                #Don't let mario die!
                '''current_lives = info['lives']
                if current_lives < previous_lives:
                    done = True
                    fitness_current = -1
                previous_lives = current_lives'''
                #reward = 100001  #This is for testing
                #This reward function gives 1 point every time xscrollLo increases
                fitness_current += reward

                '''#Anytime mario gets a powerup (score+=1000), give an extra reward
                current_score = int(info['score'])
                #if current_score - previous_score >= 1:
                if current_score > previous_score:
                    fitness_current += 25
                previous_score = current_score'''

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
                    #print('genome_id: {}, fitness_current: {}'.format(genome_id, fitness_current))
                    done = True

                genome.fitness = fitness_current

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config-feedforward'
    )

    #Load population checkpoint if one exists
    latest_checkpoint = get_latest_checkpoint(values)
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
            filename_prefix='data\\thread-{}-neat-checkpoint-'.format(values)
            )
        )

    winner = p.run(eval_genomes)

    #Save the winner
    pickle_name = 'complete_models\\winner{}.pkl'.format(values)
    with open(pickle_name, 'wb') as output:
        pickle.dump(winner, output, 1)


#helper functions
def get_latest_checkpoint(thread):
    result = None
    file_list = glob.glob('data\\thread-{}-neat-checkpoint-*'.format(thread))
    if file_list:
        max_file_num = max([int(item.replace('data\\thread-{}-neat-checkpoint-'.format(thread), '')) for item in file_list])
        result = 'data\\thread-{}-neat-checkpoint-{}'.format(thread, max_file_num)
    return result


# Necessary on Windows, but not Mac (linux?)
if __name__ == '__main__':
    p = Pool(processes=NUMBER_OF_THREADS)
    p.map(run, tuple(range(NUMBER_OF_THREADS)))
