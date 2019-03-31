# Author Alex Pimenov March 2019
# developed as a part of Reinforcement Learning learning
# BLOG  https://programmingbee.net/
# SOURCE https://github.com/Opimenov/Reinforcement_Learning/blob/master/monte_carlo_policy_eval.py
# Monte-Carlo First-Visit Policy Evaluation algorithm

################################################################################
#                              INSTRUCTIONS                                    #
################################################################################
# Open up terminal and do the following
# $ cd <into the folder where this file is>
# //this folder must also contain grid_world.py
# //https://github.com/Opimenov/Reinforcement_Learning/blob/master/grid_world.py
# $ python3 monte_carlo_policy_eval.py
################################################################################
from grid_world import *
import numpy as np

def run_episode(grid):
    '''
    Arguments:
     - grid - grid world ro run simulation on 
                     to collect a sample of returns
    Return:
     - values - dict from state tuple to scalar return value
    '''
    #since we don't want our agent to start in the same state every time,
    #otherwise in case of a deterministic policy, agent will follow only
    #a single trajectory. So randomly initialize initial state from the
    #list of available non-terminal states
    while True:  # keep setting starting state until we find non-terminal one
        start_state_index = np.random.choice(len(grid.states))
        grid.set_starting_position(grid.states[start_state_index])
        if grid.start_state != grid.terminal_states[0]:
            #0 index because we only have one terminal state
            break

    # take a note of the starting state
    initial_state = grid.current_state
    states_and_rewards = [(initial_state,0)] #list of (state,reward) tuples

    while not grid.at_terminal_state():
        reward = grid.move()
        states_and_rewards.append((grid.current_state, reward))

    # now we need to calculate returns for each state starting at the
    # last state/terminal and going backwards
    G = 0
    states_and_returns = []
    #since the value of the terminal state is zero we don't calculate
    #reward for it by ignoring it's return
    terminal_state = True
    for state, reward in reversed(states_and_rewards):
        if terminal_state:
            terminal_state = False
        else:
            states_and_returns.append((state,G))
        G = reward + grid.discount*G
    
    states_and_returns.reverse()
    return states_and_returns
    

def monte_carlo_policy_eval(simple_grid, num_of_episodes):
    '''
    simple_grid - grid world
    num_of_episodes - number of episodes that should be run to collect samples
    '''
    print("Monte-Carlo First-Vistin Policy evaluation running...")
    V = {} # Values
    G = {} # Returns
    for state in simple_grid.states:
        if simple_grid.is_terminal_state(state):
            V[state] = 0
        else:
            G[state] = []
    
    for e in range(num_of_episodes):

        states_and_returns = run_episode(simple_grid)
        visited_states = set()

        for state, ret in states_and_returns:
            if state not in visited_states:
              G[state].append(ret) # add return to all returns
              V[state]=np.mean(G[state]) #update state value by calculating mean
              visited_states.add(state)
    print("Monte-Carlo First-Vistin Policy evaluation finished")
    return V

if __name__ == '__main__':
    grid = init_simple_grid()
    # change 100 to the number of episodes you want to run
    grid.values = monte_carlo_policy_eval(grid, 1000) 
    grid.show_values()
