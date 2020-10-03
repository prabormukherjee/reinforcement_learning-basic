import numpy as np
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


def update_utility(utility_matrix, observation, new_observation, 
                   reward, alpha, gamma):
    '''Return the updated utility matrix
    @param utility_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the ste size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix
    '''
    u = utility_matrix[observation[0], observation[1]]
    u_t1 = utility_matrix[new_observation[0], new_observation[1]]
    utility_matrix[observation[0], observation[1]] += \
        alpha * (reward + gamma * u_t1 - u)
    return utility_matrix

def main():

    env = gym.make('FrozenLake8x8-v0')

    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    #Define the policy matrix
    #This is the optimal policy for world with reward=-0.04
    policy_matrix = np.array([[1,      1,  1,  -1],
                              [0, np.NaN,  0,  -1],
                              [0,      3,  3,   3]])
    print("Policy Matrix:")
    print(policy_matrix)

##    env.setStateMatrix(state_matrix)
##    env.setRewardMatrix(reward_matrix)
##    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3,4))
    gamma = 0.999
    alpha = 0.1 #constant step size
    tot_epoch = 300000
    print_epoch = 1000

    for epoch in range(tot_epoch):
        #Reset and return the first observation
        observation = env.reset()
        for step in range(1000):
            #Take the action from the action matrix
            action = policy_matrix[observation]
            print('action:',action[0])
            #Move one step in the environment and get obs and reward
            new_observation, reward, done, cheese = env.step(action)
            utility_matrix = update_utility(utility_matrix, observation, 
                                            new_observation, reward, alpha, gamma)
            observation = new_observation
            #print(utility_matrix)
            if done: break

        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:") 
            print(utility_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility_matrix)



if __name__ == "__main__":
    main()
