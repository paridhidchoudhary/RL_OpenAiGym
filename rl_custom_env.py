import gym
import os
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#Builiding an environment
#We want to get optimum shower temperature between 37 to 39 degrees


#3actions that we can take: tap up, down or leave unchanged
#action: 0= no chnage, 1= increasing the temp of shower by 1 degree, -1= decreasing the temp of shower 
#by 1 degree
class ShowerEnv(Env):
    def __init__(self):
        self.action_space=Discrete(3)
        self.observation_space= Box(low=np.array([0]), high=np.array([100]))
        #self.state= 40 + random.randint(-3,3) #initial state of shower: 40 degrees +- 3 degrees
        self.shower_length = 60
        #in step function, we are going to decrease the shower length by 1 every time we take an action
    def step(self,action):
        self.state+=action-1 # we do -1 cos we want actions as -1,0,1, discrete(3) gives 0,1,2, so by
        # subtracting 1 we can get what we want
        self.shower_length-=1

        if self.state >=37 and self.state<=39:
            reward=1
        else:
            reward=-1

        if self.shower_length<=0:
            done=True
        else:
            done=False

        info={}

        return self.state, reward, done, info
    def render(self):
        pass
    def reset(self):
        self.state= np.array([40+random.randint(-3,3)]).astype(float)
        self.shower_length=60
        return self.state
    

env= ShowerEnv()
print(env.observation_space.sample())

#test the environment
episodes=3
for episode in range(1, episodes+1):
    obs=env.reset()
    done=False
    score =0
    while not done:
        env.render()
        action= env.action_space.sample()
        obs, reward, done, info= env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

#train model

log_path=os.path.join('Training', 'Logs')
model = PPO('MlpPolicy',env, verbose=1, tensorboard_log= log_path)

model.learn(total_timesteps=4000)

shower_path= os.path.join('Training', 'Saved Models', 'Shower_Model_PPO')
model.save(shower_path)
del model
model=PPO.load(shower_path)

evaluate_policy(model,env,n_eval_episodes=10)



