# Author Alex Pimenov Jan 2019
# developed as a part of Reinforcement Learning learning
# BLOG  https://programmingbee.net/
# TODO: add github repo here
# python3.6 make sure to run sudo pip install -U future
# simple grid world to practice Reinforcement Learning 
# dynamic programming techniques
from __future__ import print_function, division
from builtins import range
import numpy as np

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
    actions - dict from state tuple to available list of
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
        
        self.rows = rows
        self.cols = cols
        self.states = []
        self.rewards = {}
        self.actions = {}
        self.action_result = {}
        self.action_probabilities = {}
        self.start_state = (0,0)
        for i in range(rows):
            for j in range(cols):
                self.states.append((i,j))
                self.terminal_states = []

    def set_discount(self, discount):
        '''sets the discount factor'''
        self.discount = discount

    def set_starting_position(self, row, col):
        ''' sets agents starting position to be (row,col)'''
        self.start_state = (row,col)
        self.current_state = (row,col)

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
        self.policy = policy

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

    def take_action(self, action):
        '''action - action from available list of actions.
        Checks if action is allowed for this state. If it is not
        does nothing. Otherwise sets current state to resulting state'''
        if action not in self.actions[current_state]:
            print("Action {0} is not allowed for this state.".format(action))
            return
        else:
            self.current_state = \
            self.action_result[(self.current_state,action)]
        
    def is_terminal_state(self, state):
        return state in self.current_state

    def show_grid(self):
        print("\t#####STATES#####")
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

    def show_rewards(self):
        print("\t#####REWARDS#####")
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
        
        

    def init_simple_grid(self):
        self.rows = 4
        self.cols = 4
        self.states = []
        for i in range(self.rows):
            for j in range(self.cols):
                self.states.append((i,j))
        self.discount = 1
        self.start_state = (0,0)
        self.terminal_states.append((self.rows-1,self.cols-1))
        #init all rewards except terminal state to be -1
        for state in self.states:
            if state in self.terminal_states:
                self.rewards[state] = 0
            else:
                self.rewards[state] = -1
    

