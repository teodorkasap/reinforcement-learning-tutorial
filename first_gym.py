import gym
env = gym.make("MsPacman-v0")

s = env.reset()
done = False

print(env.observation_space)

while not done:
    env.render()
    
    a = env.action_space.sample()
    s, r, done, info = env.step(a)

