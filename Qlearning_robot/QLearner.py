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
        self.Tc = np.zeros([self.num_states, num_actions, num_states])
        self.Tsum = np.zeros([self.num_states, num_actions, num_states])
        self.Reward = np.zeros([self.num_states,self.num_actions])
        self.experience = []
        self.dict = {}

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

        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = self.Qtable[s_prime,:].argmax()

        self.rar *=  self.radr
        self.Qtable[self.s, self.a] = (1 - self.alpha) * self.Qtable[self.s, self.a] \
                            + self.alpha * (r + self.gamma* self.Qtable[s_prime, self.Qtable[s_prime, :].argmax()])

        if self.dyna > 0:
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
            self.Tsum[self.s, self.a, :] = self.Tc[self.s, self.a, :]/self.Tc[self.s, self.a, :].sum()
            self.Reward[self.s,self.a] = (1 - self.alpha)*self.Reward[self.s, self.a] + self.alpha*r
            self.experience.append((self.s, self.a))

            for indx in range(self.dyna):
                exp = rand.choice(self.experience)
                sp = self.Tsum[exp[0], exp[1], :].argmax()
                r = self.Reward[exp[0], exp[1]]
                self.Qtable[exp[0], exp[1]] = (1 - self.alpha) * self.Qtable[exp[0], exp[1]] + \
                                                self.alpha * (r + self.gamma * self.Qtable[sp, :].max())
        self.s = s_prime
        self.a = action
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    def author(self):
        return 'zwin3'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
