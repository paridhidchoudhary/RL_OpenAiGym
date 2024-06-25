import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

environment_name='CartPole-v0'
env= gym.make(environment_name)

# env functions
# env.reset(): reset the environment and obtain initial observations
# env.render(): visualise the environment 
# env.step(action): apply an action to the environment: returns state,reward,true/false
#info
#true/false tells if the episode is complete or not
# env.close(): close down the render frame

episodes=5
for episode in range(1,episodes+1):
    state=env.reset() #env.reset() gives us the initial set of observations for our environment
    done=False
    score=0

    while not done:
        env.render() #render is used to show the environment on screen
        action=env.action_space.sample() #env has action_space consisting of discrete(2)
        # meaning 2 actions:0 1, sample() will choose randomly any action bw 0 and 1
        n_state,reward,done,info= env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))
env.close()

#Undersanding the environment: refer openai gym cartpole documentation
print(env.action_space)
print(env.observation_space) #cart angle, cart velocity, pole_angle,pole_velocity angular

#Training RL Models
#Make directories
log_path=os.path.join('Training','Logs')
print(log_path)
# def make_env():
#     return gym.make(environment_name)

# env = DummyVecEnv([make_env])
env=gym.make(environment_name)
env.observation_space.dtype = np.float32

env=DummyVecEnv([lambda: env])
# Workaround: Manually set the dtype of the observation space
#env.envs[0].observation_space.dtype = env.envs[0].observation_space.low.dtype

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000)

#save the model
# ppo_path=os.path.join('Training','Saved Models','PPO_Model_Cartpole')
# model.save(ppo_path)

# #delete model
# del model
# model.learn(total_timesteps=1000) # since model is deleted, we wont get any output

# #reload model
# model=PPO.load(ppo_path,env=env)

#evaluate policy
eval_reward,std_dev=evaluate_policy(model,env,n_eval_episodes=10,render=True)
#returns average reward and and standard deviation in that reward
#if reward is 200, then it is perfect
#cartpole reward: 1point for every 1 step that the pole remains upright for 200 timesteps)
print("Avg eval reward:", eval_reward)
env.close()

#test model
episodes=5
env=gym.make(environment_name)
for episode in range(1,episodes+1):
    
    obs=env.reset()
    done=False
    score=0

    while not done:
        env.render()
        action,_=model.predict(obs)  #model.predict returns action and next states 
        obs,reward,done,info= env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))

env.close()

#adding callback to training stage (saves the best model satisfying the set
# reward threshold)

#save the best model
env=gym.make(environment_name)
env.observation_space.dtype = np.float32
save_path=os.path.join('Training','Saved Models')
stop_callback=StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
eval_callback=EvalCallback(env,callback_on_new_best=stop_callback,eval_freq=10000
                           ,best_model_save_path=save_path
                           ,verbose=1)

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=20000,callback=eval_callback)

#changing policy
#network architecture: exampleof specifying different architecture
# for different neural networks used in PPO

net_arch=[dict(pi=[128,128,128,128],vf=[128,128,128,128])]
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path,policy_kwargs={'net_arch':net_arch})
model.learn(total_timesteps=20000,callback=eval_callback)

#using an alternate algorithm
model=DQN('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000)




