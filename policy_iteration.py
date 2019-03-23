# Author Alex Pimenov Jan 2019
# developed as a part of Reinforcement Learning learning
# BLOG  https://programmingbee.net/
# SOURCE https://github.com/Opimenov/Reinforcement_Learning/blob/master/policy_iteration.py
# Policy Iteration algorithm allows to find optimal policy/value function.

################################################################################
#                              INSTRUCTIONS                                    #
################################################################################
# Open up terminal and do the following
# $ cd <into the folder where this file is>
# //this folder must also contain grid_world.py
# //https://github.com/Opimenov/Reinforcement_Learning/blob/master/grid_world.py
# $ python3 policy_iteration.py
################################################################################
from iterative_policy_evaluation import *
from grid_world import *
import pdb 


SMALL_ENOUGH = 1e-3
GAMMA = 0.9

  
def improve_policy(action_probabilities, grid, values):
  '''
  arguments:
    action_probabilities - policy, current action probabilities.
    grid - grid of available states. Grid has a dictionary of actions.
         Where each state has an associated list of available actions.
         Grid also has a dictionary of dict from (state,action) tuple to 
         resulting state.
    values - dict of state to state values.
  returns:
    dictionary of (state,action) tuple to probability of that action  
  '''
  new_action_probs = {}
  for state in grid.states:
      #pdb.set_trace()
      best_action = ""      
      best_value = float('-inf')
      # loop through all possible actions to find the best current action
      for action in grid.state_actions[state]:
        # if action probability is 0 skip this action
        if grid.action_probabilities[(state, action)] == 0:
          continue
        value = grid.rewards[state] + GAMMA * values[grid.action_result[(state, action)]]
        # if there is a better value set action probability to 0
        if value > best_value:
          best_value = value
          best_action = action

      for action in grid.all_actions:
         if action == best_action:
           new_action_probs[(state,action)] = 1
         else:
           new_action_probs[(state,action)] = 0
  print("policy improved")
  return new_action_probs

def iterate_policy(grid, show_all_iterations=False):
  # this grid gives you a reward of -0.1 for every non-terminal state
  # we want to see if this will encourage finding a shorter path to the goal
  grid = init_simple_grid()

  # repeat until convergence - will break out when policy does not change
  best_policy_found = False
  num_of_iterations = 1
  while not best_policy_found:
    if show_all_iterations:
      print("iteration ", num_of_iterations)
      num_of_iterations = num_of_iterations + 1
    best_policy_found = True
    # policy evaluation step 
    grid.values = evaluate_policy_iteratively(grid)

    # policy improvement step    
    new_action_probabilities = improve_policy(grid.action_probabilities,
                                              grid,
                                              grid.values)
    # check if probabilities changed
    for state in grid.states:
      for action in grid.state_actions[state]:
        if new_action_probabilities[(state,action)] != \
           grid.action_probabilities[(state,action)]:
          best_policy_found = False

    grid.action_probabilities = new_action_probabilities
    grid.show_policy()

if __name__ == '__main__':
  from policy_iteration import *
  g = init_simple_grid()
  iterate_policy(g, show_all_iterations=True)

  
