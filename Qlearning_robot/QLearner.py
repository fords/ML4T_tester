"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.dyna = dyna
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.Qtable = (2 * np.random.rand(num_states, num_actions)) - 1

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.random()
        if action < self.rar:
            return rand.randint(0, self.num_actions)
        if self.verbose: print "s =", s,"a =",action
        max = -1
        for indx, val in enumerate(self.Qtable[s])
            if val > max:
                max = val
                action = indx
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        max = -1
        for indx, val in enumerate(self.Qtable[s_prime]):
            if val > max:
                max = val
                action = indx
        self.Qtable[self.s][self.a] =  (1 - self.alpha) * self.Qtable[self.s][self.a] +\
                                       self.alpha * (r + self.gamma * self.Qtable[s_prime][action])

        self.s = s_prime, self.rar *= self.radr
        action = rand.random()
        if action < self.rar:
            action =  rand.randint(0, self.num_actions)
            self.a = action
        max = -1
        for indx, val in enumerate( self.Qtable[s_prime]):
            if val > max:
                max = val
                action = indx
            self.a = action
        return action

    def author(self):
        return 'zwin3'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
