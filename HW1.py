#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:09:10 2022

@author: dicksonnkwantabisa
"""

import mdptoolbox
import numpy as np


# calcualte the number of states

run = 3


# isBadSide = np.array([1, 1, 1, 0, 0, 0])
#isBadSide = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0 , 1, 1, 0, 1, 0 ,0 , 0, 1, 0])

isBadSide = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
isGoodSide = ~isBadSide + 2
N = len(isBadSide)
states = run * N + 2 # from 0 to 2N, plus quit

Die = np.arange(1, N+1)
dollar = Die * isGoodSide

# create probability array
prob = np.zeros((2 , states, states)) # T = np.zeros((2 , states, states))
# if leave
np.fill_diagonal(prob[0], 1) # np.fill_diagonal(T[0], 1) # quit prob
# if roll
# calculate probability for input:
p = 1.0/N # 

# Create prob_1
# Create 1 X (1+run*N+2) array
zero = np.array([0]).repeat((run-1)*N+2) # Don't change it! It must have size = (run - 1)*N+1
zero_pad = [0] * ((run-1)*N+2) # to get the same shape as state_space (44 + 21=65)

isGoodSide_2 = np.concatenate((np.array([0]), isGoodSide, zero), axis=0) # rbind
new_is_good_side = np.array([0]+list(isGoodSide)+zero_pad)
# Create 1 X (run*N+3)*3 array
isGoodSide_N = np.concatenate((isGoodSide_2, isGoodSide_2), axis=0)

is_good_side_N = np.tile(new_is_good_side, ((len(new_is_good_side)),))
is_good_side_N = is_good_side_N[: (states*states)] 
is_good_side_N = is_good_side_N.reshape(states,states) # collapse into size of state space
# Create 1 X ((run*N+3)^2 array

for i in range(0, run*N+2):
    isGoodSide_N = np.concatenate((isGoodSide_N, isGoodSide_2), axis=0)
    i = i + 1
    
# Create 1 X (2N+2)^2 array by trancation
isGoodSide_N = isGoodSide_N[:(states**2)]   

isGoodSide_N = isGoodSide_N.reshape(states, states) # Reshaping (rows first)

n_actions=2
T = np.zeros((n_actions, states, states))
np.fill_diagonal(T[0], 1) # quit prob of 1
T[1] = np.triu(is_good_side_N) * (1/N) # prob of landing in good side

prob[1] = np.triu(isGoodSide_N) # upper triangle matirx
prob[1] = prob[1]*p


prob_quit = 1 - np.sum(prob[1, :states, :states-1], axis=1).reshape(-1, 1) # last column
T_prob_quit = (1 - np.sum(T[1, :states, :states-1], axis=1)).reshape(-1, 1)
T[1] = np.concatenate((T[1, :states, :states-1], T_prob_quit), axis=1) 
prob[1] = np.concatenate((prob[1, :states, :states-1], prob_quit), axis=1) #cbind
np.sum(prob[0], axis=1) # test row sum
np.sum(prob[1], axis=1) # test column sum
#print States
#print prob[0]
# Create rewards array==========================

rewards = np.zeros((2, states, states))
# if leave
rewards[0] = np.zeros((states, states))


Die = np.arange(1, N+1)
dollar = Die * isGoodSide

# if roll
# Create roll reward array
# Create 1 X (1+run*N+2) array
zero = np.array([0]).repeat((run-1)*N+2) #Don't change it! It must have size= run-1)*N+1
dollar_2 = np.concatenate((np.array([0]), dollar, zero), axis=0) # rbind



# new_is_good_side = np.array([0]+list(isGoodSide)+zero_pad)
n_sided_die = np.arange(1, N+1)
bank_roll = n_sided_die * is_good_side
bank_roll_N = np.array([0]+list(bank_roll)+zero_pad)
bank_roll_N = np.tile(bank_roll_N, ((len(bank_roll_N)),))
bank_roll_N = bank_roll_N[: (states*states)] 
bank_roll_N = bank_roll_N.reshape(states,states) # collapse into size of state space

R = np.zeros((n_actions, states, states))
R[1] = np.triu(bank_roll_N)
R_quit = -np.arange(0,states).reshape(-1,1)
R[1] = np.concatenate((R[1, :states, :states-1], R_quit), axis=1) 

# Create 1 X (run*N+3)*3 array
dollar_N = np.concatenate((dollar_2, dollar_2), axis=0)
# Create 1 X ((run*N+3)^2 array
for i in range(0, run*N+2):
    dollar_N = np.concatenate((dollar_N, dollar_2), axis=0)
    i = i + 1
# Create 1 X (2N+2)^2 array by trancation
dollar_N = dollar_N[:(states**2)]
dollar_N = dollar_N.reshape(states, states) # Reshaping (rows first)
rewards[1] = np.triu(dollar_N) # upper triangle matrix
rewards[1] = rewards[1]
rewards_quit = - np.array(range(0 ,states)).reshape(-1, 1) #convert vector to n X 1 matrix
rewards[1] = np.concatenate((rewards[1, :states, :states-1], rewards_quit), axis=1) #cbind
print (rewards[1])
vi = mdptoolbox.mdp.ValueIteration(prob, rewards, 1)
vi.run()

vi = mdptoolbox.mdp.ValueIteration(T, R, 1)
vi.run()
optimal_policy = vi.policy
expected_values = vi.V
print (optimal_policy)
print (expected_values)
print (max(expected_values))



#%% My code


is_bad_side = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])

def solve(is_bad_side):
    # get good side
    is_good_side = np.logical_not(is_bad_side).astype(int)
    roll = 3 # number of time die is rolled
    N = len(is_bad_side) # length of boolean mask
    n_sided_die = np.arange(1, N+1)
    n_actions = 2 # actions(roll or quit)
    state_space = roll * N + 2 # number of possible states
    prob = 1 / N # roll die probability of landing in a state
    
    # create empty transition (T) and reward (R) matrices
    T = np.zeros((n_actions, state_space, state_space)) 
    # T[0] <-- quit transition probabilities
    # T[1] <-- roll transition probabilities
    R = np.zeros((n_actions, state_space, state_space))
    # R[0] <-- quit reward
    # R[1] <-- roll reward    
    
    # to get the same shape as state_space (44 + 21=65)
    zero_pad = [0] * ((roll-1)*N+2) 
    new_is_good_side = np.array([0]+list(is_good_side)+zero_pad)
    is_good_side_N = np.tile(new_is_good_side, len(new_is_good_side),)
    is_good_side_N = is_good_side_N[: (state_space*state_space)] 
    # collapse into size of state space
    is_good_side_N = is_good_side_N.reshape(state_space,state_space) 
    
    # prepare rewards
    bank_roll = n_sided_die * is_good_side
    bank_roll_N = np.array([0]+list(bank_roll)+zero_pad)
    bank_roll_N = np.tile(bank_roll_N, len(bank_roll_N),)
    bank_roll_N = bank_roll_N[: (state_space*state_space)] 
    bank_roll_N = bank_roll_N.reshape(state_space,state_space) 

    # Fill T
    np.fill_diagonal(T[0], 1) # quit prob of 1
    T[1] = np.triu(is_good_side_N) * prob # prob of landing in good side
    T_quit = (1 - np.sum(T[1, :state_space, :state_space-1], axis=1)).reshape(-1, 1)
    T[1] = np.concatenate((T[1, :state_space, :state_space-1], T_quit), axis=1)

    # Fill R (we don't fill R[0], becos we don't care about quit rewards)
    R[1] = np.triu(bank_roll_N)
    R_quit = -np.arange(0,state_space).reshape(-1,1)
    R[1] = np.concatenate((R[1, :state_space, :state_space-1], R_quit), axis=1) 

    # Run Value Iteration
    # Gamma =1 because we value immediate and future rewards equal 
    gamma = 1
    epsilon = 0.01 # some threshold
    vi = mdptoolbox.mdp.ValueIteration(T, R, gamma, epsilon)    
    vi.run()
    print (np.max(vi.V))


is_bad_side= [1, 1, 1, 0, 0, 0]
solve(is_bad_side)

is_bad_side=[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
solve(is_bad_side)

is_bad_side=[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
solve(is_bad_side)



def value_iteration(self, theta, gamma, max_iter=1000):
    """
    Args:
        theta: convergence threshold
        gamma: discount rate
        max_iter: maximum number of iterations

    Returns:
        iteration: number of iterations until convergence
        pi: optimal policy
        v_values: dictionary state->v_value
    """
    v_values = {}
    for state in self.environment.get_states():
        # initialize V: for all states s, V(s) = 0
        v_values[state] = 0

    iteration, pi, deltas = self.policy_evaluation(pi=None,
                                                   v_values=v_values,
                                                   theta=theta,
                                                   gamma=gamma,
                                                   max_iter=max_iter)
    return iteration, pi, deltas, v_values


def policy_evaluation(self, pi, v_values, theta, gamma, max_iter=1000):
    iteration = 0
    deltas = []
    policy = {} if pi is None else pi
    while iteration < max_iter:
        delta = 0
        for s in v_values.keys():
            v = v_values[s]
            if pi is None:  # find best action in value iteration
                a, v_sa = self.get_optimal_action(s, v_values, gamma)
                v_values[s] = v_sa
                policy[s] = a
            else:  # evaluate the given policy
                a = pi[s]
                v_values[s] = self.calculate_v_value(s, a, v_values, gamma)
            delta = max(delta, abs(v_values[s] - v))
        iteration += 1
        deltas.append(delta)
        if delta < theta:
            break

    return iteration, policy, deltas


def get_optimal_action(self, state, v_values, gamma):
    best_v_sa = float("-inf")
    best_action = None
    for action in self.environment.get_actions(state):
        # V(s, a) for each action a on the current state s
        v_sa = self.calculate_v_value(state, action, v_values, gamma)
        # get the action a with the best V(s, a)
        if v_sa > best_v_sa:
            best_v_sa = v_sa
            best_action = action
    return best_action, best_v_sa










inspect.getsource(mdptoolbox.mdp.ValueIteration)





def solve(is_bad_side):
    
    # select the good indices
    idx = np.where(is_bad_side !=1) # 1 is bad 
    good_idx = [x+1 for x in idx[0]]
    
    N = len(is_bad_side) # size of die
    roll = 3 # number of times die is rolled
    n_actions = 2 # roll or quit
    state_space = roll * N + 2
    
    # remember that we start we $0 so 
    # potential good sides should include the first $0
    in_the_money_potential = good_idx
    in_the_money_potential.insert(0, 0)

    
    # create transition probs (T) & reward (R) matrices
    T = np.zeros((n_actions, state_space, state_space))
    R = np.zeros((n_actions, state_space, state_space))
    # T[0] <-- roll
    # T[1] <-- quit
    # R[0] <-- roll
    # R[1] <-- quit
    
    T_roll = np.zeros((state_space, state_space))
    T_quit = np.zeros((state_space, state_space))
        
    R_roll = np.zeros((state_space, state_space))
    R_quit = np.zeros((state_space, state_space))
    
    
    
    # Now, let's build the matrices T & R
    for state in range(0, state_space):
        
        # T[0][0] = np.zeros(state_space)
        # R[0][0] = np.zeros(state_space)
        
        # T_roll_row = np.zeros(state_space)
        # R_roll_row = np.zeros(state_space)
        
        
        if state in in_the_money_potential:
            terminal_prob = 1 
            
            # diversion to populate T&R for the case
            # where die roll falls in the "good place"
            for index in good_idx:
                column = state + index
                if column < state_space-1:
                    T[0][0] [column] = 1/N # prob of transitioning into that state
                    terminal_prob = terminal_prob-1/N # roll die probability of not landing in a state
                    
                    R[0][0] [column] = index # pass the reward gained when roll is on the good side
                    
                    if column not in in_the_money_potential:
                        in_the_money_potential.append(column)
                
             T[0][0] [state_space-1] = terminal_prob
             R[0][0] [state_space-1] = - state # lose everything gained
        else:
            T_roll_row[state_space-1] = 1.0
            
         T[0][state] =  T[0][0] 
         T[0][0] [state] =  R[0][0] 
        
        T[1][0][state_space-1] = 1.0 
        T[1][state] = T[1][0]
    
    # run value iteration
    vi = mdptoolbox.mdp.ValueIteration(T, R, 1.0)
    vi.run()
    
    return np.round(max(vi.V), 3)
            
                    
max_val = solve(is_bad_side=[0,0,0,0,1,1,0,1,0])




def solve(is_bad_side):
    # get the good indicies
    good_indicies = []
    for i, isBad in enumerate(is_bad_side):
        if not isBad:
            good_indicies.append(i + 1)
    print(good_indicies)
    
    N = len(is_bad_side)
    max_states_n = 3 * N + 2

    T = np.zeros((2, max_states_n, max_states_n))
    
    T_roll = np.zeros((max_states_n, max_states_n))
    T_quit = np.zeros((max_states_n, max_states_n))
    
    R = np.zeros((2, max_states_n, max_states_n))
    
    R_roll = np.zeros((max_states_n, max_states_n))
    R_quit = np.zeros((max_states_n, max_states_n))
    
    possible_rows = [0] + good_indicies
    print(possible_rows)
    
    # build T and R matricies
    for row in range(0, max_states_n):
        # row vector
        T_roll_row = np.zeros(max_states_n)# T[0][0]
        R_roll_row = np.zeros(max_states_n)# R[0][0]
        
        if row in possible_rows:
            terminal_p = 1 # take money a quit game probability
            for idx in good_indicies:
                col = idx + row
                if col < max_states_n - 1:
                    T_roll_row[col] = 1 / N # prob = 1 / N # roll die probability of landing in a state
                    terminal_p = terminal_p - (1 / N) #prob =1 - 1 / N # roll die probability of not landing in a state
                    
                    R_roll_row[col] = idx
                    
                    # if col not in possible_rows:
                    #     possible_rows.append(col)
            T_roll_row[max_states_n - 1] = terminal_p # # T[0][0]
            R_roll_row[max_states_n - 1] = -row #  R[0][0]
        else:
            T_roll_row[max_states_n - 1] = 1.0 # T[0][0]
        
        T_roll[row] = T_roll_row # T[0][row]
        R_roll[row] = R_roll_row # R[0][row]
        
        T_quit_row = np.zeros(max_states_n) # T[1][0]
        T_quit_row[max_states_n - 1] = 1.0 # T[1][0][max_states_n-1]
        T_quit[row] = T_quit_row # T[1][row]
        
    #print(R_roll)
    T[0] = T_roll
    T[1] = T_quit
    R[0] = R_roll
    R[1] = R_quit # R[1][0] no need becasue we don't care about quitting rewards
    
    vi = mdptoolbox.mdp.ValueIteration(T, R, 1.0)
    vi.run()
    
    return np.round(max(vi.V), 4)


max_val = solve(is_bad_side=[0,0,0,0,1,1,0,1,0])



















