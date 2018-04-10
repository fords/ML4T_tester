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
        self.q_table = (2 * np.random.rand(num_states, num_actions)) - 1

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        random_number = rand.random()
        if random_number < self.rar:
            return rand.randint(0, self.num_actions)

        max_value = -float("inf")
        max_index = None
        for index, value in enumerate(self.q_table[s]):
            if value > max_value:
                max_value = value
                max_index = index
        return max_index

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        max_value = -float("inf")
        max_index = None
        for index, value in enumerate(self.q_table[s_prime]):
            if value > max_value:
                max_value = value
                max_index = index
        self.q_table[self.s][self.a] = \
            (1 - self.alpha) * self.q_table[self.s][self.a] + \
            self.alpha * (r + self.gamma * self.q_table[s_prime][max_index])

        self.s = s_prime
        random_number = rand.random()
        if random_number < self.rar:
            action = rand.randint(0, self.num_actions)
            self.a = action

        self.rar *= self.radr
        max_value = -float("inf")
        max_index = None
        for index, value in enumerate(self.q_table[s_prime]):
            if value > max_value:
                max_value = value
                max_index = index
        self.a = max_index
        return max_index

    def author(self):
        return 'zwin3'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
