# Author Alex Pimenov Jan 2019
# developed as a part of Reinforcement Learning learning
# BLOG  https://programmingbee.net/
# SOURCE https://github.com/Opimenov/Reinforcement_Learning/blob/master/grid_world.py
# python3.6 make sure to run sudo pip install -U future
# simple grid world to practice Reinforcement Learning 
# dynamic programming techniques
from __future__ import print_function, division
from builtins import range
import numpy as np

# this grid world is deterministic, in a sense that each
# allowed action has definitive result. There is no environment
# acting upon our agent.
class Grid_World(object): #environment
    '''
    Grid world implementation for exercises from Sutton RL book.
    state - is a tuple (row,col)
    FIELDS:
    rows - number of rows in the drig 
    cols - number of columns in the drig  
    states - list of (row,col) tuples
    start_state - agents starting position
    terminal_states - list of terminal states
    current_state - state of the agent
    rewards - dict from state tuple to scalar reward
    all_actions - list of all available actions
    state_actions - dict from state tuple to available list of
              actions for that state
    action_results - dict from (state,action) tuple to resulting state
    TODO:: change action result to return a list of tuples
           (future_state, transition_probability)
    action_probabilities - dict from (state,action) to probability
                           of this action being taken by the agent.
                           Could say, that this is a policy.
    '''
    
    def __init__(self, rows, cols):
        ''' args:
        rows - number of rows in the grid
        cols - number of columns in the grid
        DESCRIPTION:
        rows*cols gives us the state space.Used to initialize a list of
        available states. 
        Initializes a list of states.
        If create_simple_grid is true: creates simple 4x4
        grid world with uniform
        policy and UDLR moves, (0,0) starting state, 
        (rows-1,cols-1) terminal state, discount of 1,
        deterministic state transition after action
        and each move is penilized with -1 reward.
        Reward for leaving each state is -1, for
        terminal state is 0'''
        self.values = {}
        self.rows = rows
        self.cols = cols
        self.states = []
        self.rewards = {}
        self.all_actions = []
        self.state_actions = {}
        self.action_result = {}
        self.action_probabilities = {} # policy
        self.start_state = (0,0)
        self.terminal_states = []        
        for i in range(rows):
            for j in range(cols):
                self.states.append((i,j))
        self.current_state = (0,0)

    def set_discount(self, discount):
        '''sets the discount factor'''
        self.discount = discount

    def set_starting_position(self, state):
        ''' sets agents starting position to be state(row,col)'''
        self.start_state = state
        self.current_state = state

    def add_terminal_state(self, row,col):
        ''' adds terminal position row,col to the list of 
        terminal states'''
        self.terminal_states.append((row,col))

    def set_rewards(self, rewards):
        '''rewards - dict from (row,col) to reward '''
        self.rewards = rewards

    def set_actions(self, actions):
        '''actions - list of available actions'''
        self.actions = actions

    def set_policy(self, policy):
        '''policy - dict from (state,action) tuple to probability'''
        self.action_probabilities = policy

    def set_action_results(self, action_result):
        '''action_result - dict from (state,action)
        to resulting state'''
        self.action_result = action_result

    def get_list_of_actions_for_state(self, state):
        '''retuns a list of actions for a given state'''
        return self.actions[state]

    def set_current_state(self, state):
        '''state is a tuple (row,col)
        sets current state of the env to be state'''
        self.current_state = state

    def current_state(self):
        '''returns current state of the env'''
        return self.current_state            
        
    def is_terminal_state(self, state):
        return state in self.terminal_states

    def at_terminal_state(self):
        return self.current_state in self.terminal_states

    def show_grid(self):
        print("#####STATES#####")
        for row in range(self.rows):
            for col in range(self.cols):
                print("+------",end="")
            print("+")
            for col in range(self.cols):
                print("|{0}".format((row,col)),end="")
            print("|")
        for col in range(self.cols):
            print("+------",end="")
        print("+")
        print("STARTING STATE {0}".format(self.start_state))
        print("TERMINAL STATES : ",end="")
        for state in self.terminal_states:
            print(state,end="  ")
        print("")

    def show_rewards(self):
        print("#####REWARDS#####")
        for row in range(self.rows):
            for col in range(self.cols):
                print("+---",end="")
            print("+")
            for col in range(self.cols):
                print("|{:>3}".format(self.rewards[(row,col)]),end="")
            print("|")
        for col in range(self.cols):
            print("+---",end="")
        print("+")

    def str_list(self, list):
        s = ""
        for t in list: s = s+t
        return s
        
    def show_policy(self):
        print("#####POLICY#####")
        for row in range(self.rows):
            for col in range(self.cols):
                print("+----",end="")
            print("+\n|",end="")
            for col in range(self.cols):
                actions = []
                for action in self.state_actions[(row,col)]:
                    #show only if action probability is greater than 0
                    if self.action_probabilities[((row,col),action)] != 0:
                        actions.append(action)
                print("{:>4}|".format(self.str_list(actions)),end="")
            print("")
        for col in range(self.cols):
            print("+----",end="")
        print("+")

    def str_dict(self,pair):
        '''action probability pair'''
        return ""+str(pair[0])+":"+str(pair[1])
        
    def show_all_action_probabilities(self):
        print("#####ACTION PROBABILITIES#####")
        for row in range(self.rows):
            print("+-------------------------------------+")            
            for col in range(self.cols):
                act_list = self.state_actions[(row,col)]
                print("state:"+self.str_dict((row,col)))
                for a in act_list:
                    print(" {:>4} ".format(\
                        self.str_dict(\
                                      (a,self.action_probabilities[(row,col),a])\
                        )\
                    ),end="")
                print("")
            print("")

    def show_values(self):
        if not self.values: # values are empty, get out
            return
        '''values is a dict from (row,col) to scalar value'''
        print("#####VALUES#####")
        for row in range(self.rows):
            for col in range(self.cols):
               print("+------",end="")
            print("+\n|",end="")
            for col in range(self.cols):
               value = self.values[(row,col)]
               if value >= 0:
                   print("{:>6.2f}|".format(value),end="")
               else:
                   print("{:>6.2f}|".format(value),end="")
            print("")
        for col in range(self.cols):
            print("+------",end="")
        print("+")

    def show_deterministic_policy(self):
        print("#####BEST ACTIONS#####")
        for row in range(self.rows):
            for col in range(self.cols):
                print("+----",end="")
            print("+\n|",end="")
            for col in range(self.cols):
                print("{:>4}|".format(self.str_list(self.state_actions[(row,col)])),end="")
            print("")
        for col in range(self.cols):
            print("+----",end="")
        print("+")

        
    def show_env(self):
        self.show_grid()
        self.show_rewards()
        self.show_all_action_probabilities()
        self.show_policy()
        self.show_values()

    def take_action(self, action):
        '''
        Checks if action is allowed for this state. If it is not
        does nothing. Otherwise sets current state to resulting state
        Arguments:
           action - action from available list of actions.
        Return:
           reward - reward for leaving current state, as a result of the action
        '''
        reward = 0
        if action not in self.state_actions[self.current_state]:
            print("Action {0} is not allowed for this state.".format(action))
        else:
            reward = self.rewards[self.current_state]
            self.current_state = self.action_result[(self.current_state,action)]
        return reward


    #since we want our policy to be non-deterministic we are goint to
    #use the following function to pick action with some probability
    #np.random.choice(
    #         ['UP', 'DOWN', 'LEFT', 'RIGHT'],
    #         1  # we are only picking one element
    #       p=[ 0.5,   0.1,    0.1,    0.3])
    # actual prob values will be different
    def select_action(self):
        s = self.current_state
        acts = []
        probs = []
        for a in self.state_actions[self.current_state]:
           # this will make sure that values are lined up
           acts.append(a)
           probs.append(self.action_probabilities[(s,a)])

        #choice returns an array with a single element, so [0]
        return np.random.choice(acts, 1, probs)[0]

                        
    def move(self):
        '''
        Return reward that we got from taking an action
        '''
        return self.take_action(self.select_action())
        
        
# end of Grid_World class

def init_simple_grid():
    g = Grid_World(4,4)
    #init all states
    g.discount = 1
    #set starting and terminal states
    g.start_state = (0,0)
    g.terminal_states.append((g.rows-1,g.cols-1))
    #init all rewards except terminal state to be -1
    for state in g.states:
        if state in g.terminal_states:
            g.rewards[state] = 0
        else:
            g.rewards[state] = -1
    #init all_actions
    U = "\u2191"
    R = "\u2192"
    D = "\u2193"
    L = "\u2190"
    #init state_actions
    g.all_actions = [U,R,D,L]
    g.state_actions[(0,0)]=[R,D]
    g.state_actions[(0,1)]=[R,D,L]
    g.state_actions[(0,2)]=[R,D,L]
    g.state_actions[(0,3)]=[D,L]
    g.state_actions[(1,0)]=[U,R,D]
    g.state_actions[(1,1)]=[U,R,D,L]
    g.state_actions[(1,2)]=[U,R,D,L]
    g.state_actions[(1,3)]=[U,D,L]
    g.state_actions[(2,0)]=[U,R,D]
    g.state_actions[(2,1)]=[U,R,D,L]
    g.state_actions[(2,2)]=[U,R,D,L]
    g.state_actions[(2,3)]=[U,D,L]
    g.state_actions[(3,0)]=[U,R]
    g.state_actions[(3,1)]=[U,R,L]
    g.state_actions[(3,2)]=[U,R,L]
    g.state_actions[(3,3)]=[]
    #init action probabilities. The policy
    #this is agly but easy to do and understand
    #row 0
    g.action_probabilities[((0,0),R)]=0.5
    g.action_probabilities[((0,0),D)]=0.5
    g.action_probabilities[((0,0),U)]=0
    g.action_probabilities[((0,0),L)]=0
    g.action_probabilities[((0,1),R)]=0.33
    g.action_probabilities[((0,1),D)]=0.33
    g.action_probabilities[((0,1),L)]=0.33
    g.action_probabilities[((0,1),U)]=0
    g.action_probabilities[((0,2),R)]=0.33
    g.action_probabilities[((0,2),D)]=0.33
    g.action_probabilities[((0,2),L)]=0.33
    g.action_probabilities[((0,2),U)]=0
    g.action_probabilities[((0,3),D)]=0.5
    g.action_probabilities[((0,3),L)]=0.5
    g.action_probabilities[((0,3),U)]=0
    g.action_probabilities[((0,3),R)]=0   
    #row 1
    g.action_probabilities[((1,0),L)]=0    
    g.action_probabilities[((1,0),U)]=0.33
    g.action_probabilities[((1,0),R)]=0.33
    g.action_probabilities[((1,0),D)]=0.33
    g.action_probabilities[((1,1),U)]=0.25
    g.action_probabilities[((1,1),R)]=0.25
    g.action_probabilities[((1,1),D)]=0.25
    g.action_probabilities[((1,1),L)]=0.25
    g.action_probabilities[((1,2),U)]=0.25
    g.action_probabilities[((1,2),R)]=0.25
    g.action_probabilities[((1,2),D)]=0.25
    g.action_probabilities[((1,2),L)]=0.25
    g.action_probabilities[((1,3),U)]=0.33
    g.action_probabilities[((1,3),D)]=0.33
    g.action_probabilities[((1,3),L)]=0.33
    g.action_probabilities[((1,3),R)]=0    
    #row 2
    g.action_probabilities[((2,0),L)]=0    
    g.action_probabilities[((2,0),U)]=0.33
    g.action_probabilities[((2,0),R)]=0.33
    g.action_probabilities[((2,0),D)]=0.33
    g.action_probabilities[((2,1),U)]=0.25
    g.action_probabilities[((2,1),R)]=0.25
    g.action_probabilities[((2,1),D)]=0.25
    g.action_probabilities[((2,1),L)]=0.25
    g.action_probabilities[((2,2),U)]=0.25
    g.action_probabilities[((2,2),R)]=0.25
    g.action_probabilities[((2,2),D)]=0.25
    g.action_probabilities[((2,2),L)]=0.25
    g.action_probabilities[((2,3),U)]=0.33
    g.action_probabilities[((2,3),D)]=0.33
    g.action_probabilities[((2,3),L)]=0.33
    g.action_probabilities[((2,3),R)]=0    
    #row 3
    g.action_probabilities[((3,0),L)]=0
    g.action_probabilities[((3,0),D)]=0    
    g.action_probabilities[((3,0),U)]=0.5
    g.action_probabilities[((3,0),R)]=0.5
    g.action_probabilities[((3,1),D)]=0        
    g.action_probabilities[((3,1),U)]=0.33
    g.action_probabilities[((3,1),R)]=0.33
    g.action_probabilities[((3,1),L)]=0.33
    g.action_probabilities[((3,2),D)]=0        
    g.action_probabilities[((3,2),U)]=0.33
    g.action_probabilities[((3,2),R)]=0.33
    g.action_probabilities[((3,2),L)]=0.33
    g.action_probabilities[((3,3),D)]=0        
    g.action_probabilities[((3,3),U)]=0
    g.action_probabilities[((3,3),R)]=0
    g.action_probabilities[((3,3),L)]=0
    #init action_result dict
    #row 0
    g.action_result[((0,0),R)]=(0,1)
    g.action_result[((0,0),D)]=(1,0)
    g.action_result[((0,0),U)]=(0,0)
    g.action_result[((0,0),L)]=(0,0)
    g.action_result[((0,1),R)]=(0,2)
    g.action_result[((0,1),D)]=(1,1)
    g.action_result[((0,1),L)]=(0,0)
    g.action_result[((0,2),R)]=(0,3)
    g.action_result[((0,2),D)]=(1,2)
    g.action_result[((0,2),L)]=(0,1)
    g.action_result[((0,3),D)]=(1,3)
    g.action_result[((0,3),L)]=(0,2)
    g.action_result[((0,3),U)]=(0,3)
    g.action_result[((0,3),R)]=(0,3)
    #row 1
    g.action_result[((1,0),L)]=(1,0)
    g.action_result[((1,0),U)]=(0,0)
    g.action_result[((1,0),R)]=(1,1)
    g.action_result[((1,0),D)]=(2,0)
    g.action_result[((1,1),U)]=(0,1)
    g.action_result[((1,1),R)]=(1,2)
    g.action_result[((1,1),D)]=(2,1)
    g.action_result[((1,1),L)]=(0,1)
    g.action_result[((1,2),U)]=(0,2)
    g.action_result[((1,2),R)]=(1,3)
    g.action_result[((1,2),D)]=(2,2)
    g.action_result[((1,2),L)]=(1,1)
    g.action_result[((1,3),U)]=(0,3)
    g.action_result[((1,3),D)]=(2,3)
    g.action_result[((1,3),L)]=(1,2)
    g.action_result[((1,3),R)]=(1,3)
    #row 2
    g.action_result[((2,0),L)]=(2,0)
    g.action_result[((2,0),U)]=(1,0)
    g.action_result[((2,0),R)]=(2,1)
    g.action_result[((2,0),D)]=(3,0)
    g.action_result[((2,1),U)]=(1,1)
    g.action_result[((2,1),R)]=(2,2)
    g.action_result[((2,1),D)]=(3,1)
    g.action_result[((2,1),L)]=(2,2)
    g.action_result[((2,2),U)]=(1,2)
    g.action_result[((2,2),R)]=(2,3)
    g.action_result[((2,2),D)]=(3,2)
    g.action_result[((2,2),L)]=(2,1)
    g.action_result[((2,3),U)]=(1,3)
    g.action_result[((2,3),D)]=(3,3)
    g.action_result[((2,3),L)]=(2,2)
    g.action_result[((2,3),R)]=(2,3)
    #row 3
    g.action_result[((3,0),L)]=(3,0)
    g.action_result[((3,0),D)]=(3,0)    
    g.action_result[((3,0),U)]=(2,0)
    g.action_result[((3,0),R)]=(3,1)
    g.action_result[((3,1),D)]=(3,1)     
    g.action_result[((3,1),U)]=(2,1)
    g.action_result[((3,1),R)]=(3,2)
    g.action_result[((3,1),L)]=(3,0)
    g.action_result[((3,2),D)]=(3,2)        
    g.action_result[((3,2),U)]=(2,2)
    g.action_result[((3,2),R)]=(3,3)
    g.action_result[((3,2),L)]=(3,1)
    g.action_result[((3,3),D)]=(3,3)
    g.action_result[((3,3),U)]=(3,3)
    g.action_result[((3,3),R)]=(3,3)
    g.action_result[((3,3),L)]=(3,3)
    #uncomment to display detailed grid debuggin info
    #g.show_env()
    return g
