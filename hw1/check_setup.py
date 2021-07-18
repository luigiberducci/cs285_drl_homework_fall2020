import gym
from gym.wrappers import Monitor

env = gym.make("Ant-v2")
env = Monitor(env, 'videos/check_setup', force=True)
observation = env.reset()

for i in range(100):
    print(i)
    env.render(mode='rgb_array')
    obs, rew, term, _ = env.step(env.action_space.sample())
    if term:
        break
env.close()
