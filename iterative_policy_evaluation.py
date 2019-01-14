# Author Alex Pimenov Jan 2019
# developed as a part of Reinforcement Learning learning
# BLOG  https://programmingbee.net/
# SOURCE https://github.com/Opimenov/Reinforcement_Learning/blob/master/iterative_policy_evaluation.py
# python3.6 make sure to run sudo pip install -U future
# Iterative Policy Evaluation algorithm
from grid_world import init_simple_grid

def show_values(values, grid, iteration):
    '''values is a dict from (row,col) to scalar value'''
    print("#####VALUES##### iteration {0}".format(iteration))
    for row in range(grid.rows):
        for col in range(grid.cols):
            print("+------",end="")
        print("+\n|",end="")
        for col in range(grid.cols):
            value = values[(row,col)]
            if value >= 0:
                print(" {:>5.2f}|".format(value),end="")
            else:
                print("{:>5.2f}|".format(value),end="")
        print("")
    for col in range(grid.cols):
        print("+------",end="")
    print("+")
    

def run_algorithm():
    '''
    num_of_states - number of available states
    actions - matrix where col num represents a state, row represents action, 
              element at action,state is the resulting state 
              (0 if move does not change state)
    transition_matrix - probabilities of state transitions for a given policy
                        num of states x num of states
    rewards - an array where index is a state and a value is a reward for
              leaving that state
    discount - value 0..1
    policy - matrix probability distribution of actions over states
              num of states x num of actions. Each row contains probabilities
              of taking some action while being in row state
    '''
    grid = init_simple_grid()
    states = grid.states
    gamma = grid.discount
    rewards = grid.rewards
    actions = grid.state_actions
    action_result = grid.action_result
    action_prob = grid.action_probabilities
    threshold = 0.00000000001
    #initialize dictionary of state to values 
    values = {}
    #set initial values to be zeros
    for s in states: values[s] = 0
    show_values(values,grid, 0)
    #Repeat until convergence
    iteration = 0
    while True:
        delta = 0
        for state in states:
            old_value = values[state]
            #exclude terminal states.
            #there are no actions defined for terminal state
            if len(grid.state_actions[state])!=0: 
                weighted_sum = 0
                #accumulate weighted with probabilities values of possible next states
                for action in actions[state]:
                    prob = action_prob[(state,action)]
                    next_state_value = values[action_result[(state,action)]]
                    weighted_sum = weighted_sum + (prob * next_state_value)
                new_value = rewards[state] + gamma * weighted_sum
                values[state] = new_value
                delta = max(delta, abs(old_value - new_value))
        iteration = iteration + 1
        show_values(values,grid,iteration)
        if delta < threshold:
            print("policy evaluated")
            break
#    return values
    
