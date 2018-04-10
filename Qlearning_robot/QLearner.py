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
        self.Qtable = np.zeros(shape=(num_states, num_actions))
        self.R = np.zeros(shape=(num_states, num_actions))
        self.T = {}
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        rand_int = rand.random()
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.Qtable[s, :].argmax()
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.Qtable[self.s, self.a] = (1 - self.alpha) * self.Qtable[self.s, self.a] \
                                    + self.alpha * (r + self.gamma* self.Qtable[s_prime, self.Qtable[s_prime, :].argmax()])
        self.rar *= self.radr
        if self.dyna > 0:

            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] \
                                        + self.alpha * r

            if (self.s, self.a) in self.T:
                if s_prime in self.T[(self.s, self.a)]:
                    self.T[(self.s, self.a)][s_prime] += 1
                else:
                    self.T[(self.s, self.a)][s_prime] = 1
            else:
                self.T[(self.s, self.a)] = {s_prime: 1}

            Q = deepcopy(self.Qtable)
            for i in range(self.dyna):
                s = rand.randint(0, self.num_states - 1)
                a = rand.randint(0, self.num_actions - 1)
                if (s, a) in self.T:
                    # Find the most common s_prime as a result of taking a in s
                    s_pr = max(self.T[(s, a)], key=lambda k: self.T[(s, a)][k])
                    # Update the temporary Q table
                    Q[s, a] = (1 - self.alpha) * Q[s, a] \
                                + self.alpha * (self.R[s, a] + self.gamma
                                * Q[s_pr, Q[s_pr, :].argmax()])

            self.Qtable = deepcopy(Q)


        action  = self.query_set_state(s_prime)
        return action


    def author(self):
        return 'zwin3'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
