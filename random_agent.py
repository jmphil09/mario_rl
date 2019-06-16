import retro


def main():
    env = retro.make(game='SuperMarioBros-Nes', state='Level2-1.state')
    obs = env.reset()
    done = False
    while not done:
        #action will be given by the neural net
        action = env.action_space.sample()
        action = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        obs, reward, done, info = env.step(action)
        env.render()
        print('obs.shape: {}, reward: {}, done: {}, info: {}, action: {}'.format(obs.shape, reward, done, info, action))
    env.close()


if __name__ == "__main__":
    main()

'''
import retro
print(retro.data.list_games())

python -m retro.import .
'''
