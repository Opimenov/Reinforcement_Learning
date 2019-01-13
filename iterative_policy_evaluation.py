import numpy as np

def evaluate_policy_iteratively(num_of_states,
                                actions,
                                transition_matrix,
                                rewards,
                                discount,
                                policy,
                                threshold):
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
    #initialize an array V(s)=0, for all states
    values = np.zeros(num_of_states)
    #Repeat
    while True:
        biggest_delta = 0
        for state in range(1,num_of_states+1):
            old_value = values[state]
            #for each action
            for probability in policy[:,state]:
                values[state] += rewards[state] + dicount *    # sum of entries( row of policy[state]  * 
        
evaluate_policy_iteratively(5,1,1,1,1)                      

