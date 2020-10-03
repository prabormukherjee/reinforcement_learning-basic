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

level= [0,0,0,0,0,0,0]

# generate a random
def create_random_policy(env):
     policy = {}
     for key in range(0, env.observation_space.n):
          current_end = 0
          p = {}
          for action in range(0, env.action_space.n):
               p[action] = 1 / env.action_space.n
          policy[key] = p
     return policy
##pi = np.random.dirichlet(np.ones(4), size=16)
##pi.shape



def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
         Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q
def run_game(env, policy, owner,display=False):
##    level[1]+=1
    print(owner)
    env.reset()
    episode = []
    finished = False
    while not finished:
        fin = 0
        s = env.env.s
        if display:
            clear_output(True)
            env.render()
            sleep(1)
        timestep = []
        timestep.append(s)
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
            top_range += prob[1]
            if n < top_range:
                action = prob[0]
                #print("Action = ", action)
                break

        state, reward, finished, info = env.step(action)
        timestep.append(action)
        timestep.append(reward)
        if finished==True:
            fin=1
        timestep.append(fin)
        episode.append(timestep)
        if display:
            clear_output(True)
            env.render()
            sleep(.1)
    
    return episode


def test_policy(policy, env):
      wins = 0
      r = 100
      #print(policy)
      for i in range(r):
          w = run_game(env, policy,2)[-1][-1]
          print("win?:",w)
          if w == 1:
              wins += 1
      return wins / r


def td_predict(env, policy= None):
    if not policy:
        policy = create_random_policy(env)
    
    #v = collections.defaultdict(float)
    v = np.random.dirichlet(np.ones(0), size=16)
        
     #pi = create_random_policy(env)
    nb_episodes = 50
    alpha = 0.05
    gamma = 0.9
    S = env.reset()
##    env.action_space
##    env.observation_space
##    env.reset()
##    level[2]+=1

##     env.step(0)
##     env.step(1)
    assert 0 < alpha <= 1
    #done = False
    #v = create_state_action_dictionary(env, policy)
    for _ in range(nb_episodes):
        done = False
        reward=0
        #S = env.reset()
        #S = env.env.s
        episode=run_game(env, policy,1)
        print("ep",episode)
        ##print('policy:',policy)
        #print(S)
         
        returns = {}
        
##         level[3]+=1
##         print(level)
         # sample an action from the prob distribution of the observation 
         
##             action = np.random.choice(action[observation])
         #print(action)
        for i in reversed(range(0, len(episode))):
            
            
##             level[4]=i
##             print(level)
            #action = np.random.choice(4, 1, p=episode[S])
            #print('action:',action)
            
            s_t, a_t, r_t, fin = episode[i]
            #new_observation, rewardf, done, info = env.step(S)#New state observation, new action, step reward
            #print('Step reward:',episode[i])
##            if(r_t!=rewardf):
##                
##                print(new_observation, rewardf, done, info)
##                print(s_t, a_t, r_t, fin)
##                print("Noooooooo!!!!!!!")
##                break
            state_action_pair = (s_t, a_t)
            reward+= r_t #add step reward to total
            ##value of a previous state = value of previous state + learning_rate * (reward + discount_factor(value of current state) — value of previous state)
             
            
                
                #v[s_t][a_t]=sum(returns[state_action_pair]) / len(returns[state_action_pair])
            v[S]= v[S] + alpha * (reward + gamma * v[s_t] - v[S])
            #v[s_t][1] +=  alpha * (abs(sum(v[s_t].values())) + (gamma * (abs(sum(v[s_t].values())) - abs(sum(v[S])))))
            #policy[S][1] += \
                 #alpha * (r_t + (gamma * v[s_t])-v[S])
            v_list = list(map(lambda x: x[1], v[s_t].items()))
            indices = [i for i, x in enumerate(v_list) if x == max(v_list)]
            max_Q = random.choice(indices)
            A_star = v[s_t] # 14.
             
            #print('S:',S)
            vcopy = v[S]
            newv = v[s_t]
            for a in policy[s_t].items():
                if a[0] == A_star:
##                        v[s_t][a_t]= v[s_t][a_t] + alpha * (reward + gamma * v[s_t][a_t] - v[s_t][a_t])
                    policy[s_t][a[0]] = 1-  alpha * (abs(sum(v[s_t].values())) + (gamma * (abs(sum(newv.values())) - abs(sum(vcopy)))))
                else:
##                        v[s_t][a_t]=  alpha * (reward + gamma * v[s_t][a_t] - v[s_t][a_t])
                    policy[s_t][a[0]] = alpha * (abs(sum(v[s_t].values())) + (gamma * (abs(sum(newv.values())) - abs(sum(vcopy)))))
            
            S=s_t
            if fin==1:
                done=True
            if(done):
                break
    ##print('policy:',policy)
    print('Q value:',v[S])
    print('Total ep. reward:',reward)
    return v
##level[0]+=1
##print(level)
env = gym.make('FrozenLake8x8-v0')
policy= td_predict(env)
print(test_policy(policy,env))
