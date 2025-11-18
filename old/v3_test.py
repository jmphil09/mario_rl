'''from GameRunner_v3 import GameRunner


runner = GameRunner(
    num_threads=32,
    show_game=True,
    show_nn_view=False,
    level_end_score=3186,
    convolution_weight=8,
    config_file_name='config',
    worker_start_num=0,
    max_generation=200,
    data_folder='data'
)

#runner.run_all_threads()
runner.run_in_parallel()
'''


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

def one_hot_decode(ls):
    return ls.index(max(ls))

def run_one_episode(level, worker_num):
    env = gym_super_mario_bros.make(level)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    obs = env.reset()
    env.action_space.sample()

    input_x, input_y, input_colors = env.observation_space.shape

    net = None #This will be the NN

    current_max_fitness = 0
    fitness_current = 0
    frame = 0
    frame_counter = 0

    done = False

    while not done:

        #env.render()
        frame += 1

        obs = cv2.resize(obs, (input_x, input_y))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = np.reshape(obs, (input_x, input_y))

        nn_output = one_hot_decode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        obs, reward, done, info = env.step(nn_output)

        fitness_current += reward

        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            frame_counter = 0
        else:
            frame_counter += 1

        if done or frame_counter == 250:
            done = True


worker_levels = ['SuperMarioBros-' + str(world) + '-' + str(level) + '-v0' for world in range(1, 9) for level in range(1, 5)]
#level = 'SuperMarioBros-1-1-v0'
#run_one_episode(level, 0)


def run_multiple_episodes(levels):
    num_cores = 32
    p = Pool(processes=num_cores)
    worker_range = range(1, num_cores+1)
    worker_levels = ['SuperMarioBros-' + str(world) + '-' + str(level) + '-v0' for world in range(1, 9) for level in range(1, 5)]
    print(tuple(zip(worker_levels, worker_range)))
    p.starmap(run_one_episode, [[worker_levels[i], i] for i in range(0, num_cores)])


run_multiple_episodes(worker_levels)
