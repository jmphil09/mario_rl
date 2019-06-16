import retro


env = retro.make(game='SuperMarioBros-Nes', state='Level1-1.state')
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('obs.shape: {}, reward: {}, done: {}, info: {}, action: {}'.format(obs.shape, reward, done, info, action))
env.close()
