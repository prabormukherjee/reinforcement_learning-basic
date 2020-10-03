import numpy as np
import matplotlib.pyplot as plt
import gym
import operator
from IPython.display import clear_output
from time import sleep
import random
import itertools
import tqdm
import collections

tqdm.monitor_interval = 0


env = gym.make('FrozenLake-v0')
epsilon = 0.9
nb_episodes = 100000
max_steps = 100
alpha = 0.85
gamma = 0.95
  
#Initializing the Q-matrix 
Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state): 
    action=0
##    if np.random.uniform(0, 1) < epsilon: 
##        action = env.action_space.sample() 
##    else: 
##        action = np.argmax(Q[state, :])
    values = Q[state, :]
    max_value = max(values)
    actions = [a for a in range(len(values))]
    greedy_actions = [a for a in range(len(values)) if values[a] == max_value]
    if (random.random() < epsilon):
        return random.choice(actions)
    else:
        return random.choice(greedy_actions)
    
    #return action



def update(state, state2, reward, action, action2): 
    predict = Q[state, action] 
    target = reward + gamma * Q[state2, action2] 
    Q[state, action] = Q[state, action] + alpha * (target - predict) 


reward=0


def test_policy(policy, env):
      wins = 0
      r = 100
          
      perf = run_game(env, policy, r)
      if perf > 1:
          wins += 1
      return wins / r


  
# Starting the SARSA learning
def run_game(env, Q, nb_episodes):
    total_score = 0
    for episode in range(nb_episodes):
        t = 0
        S = env.reset() 
        A1 = choose_action(S)
        score = 0
        while t < max_steps: 
            #Visualizing the training 
            #env.render() 
              
            #Getting the next state 
            S2, reward, done, info = env.step(A1) 
            #Choosing the next action 
            A2 = choose_action(S2) 
              
            #Learning the Q-value 
            update(S, S2, reward, A1, A2) 
      
            S = S2 
            A1 = A2 
              
            #Updating the respective vaLues 
            t += 1
            score += reward
            total_score += reward
              
            #If at the end of learning process 
            if done:
                if(score>1):
                    print("Perf: ",score)
                break
            
    print ("Performance : ", total_score/nb_episodes/100)
    return total_score


run_game(env, Q, nb_episodes)

print(test_policy(Q,env))
#Visualizing the Q-matrix 
#print(Q) 
