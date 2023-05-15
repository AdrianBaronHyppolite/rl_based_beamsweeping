from operator import truediv
from gym import Env
from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2
from gym.spaces import Discrete, Box
from antennaOG import AntennaModel
import antennaOG
import pandas as pd
import fortress
import math
import numpy as np
import random
import gym
import time


class RLIAEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.reward = 0
        self.done = False
        self.state = 0
        # self.observation_space = Box(low=np.array([-100, -100]), high=np.array([100, 100]), dtype=np.float32)
        self.observation_space = Box(low=-100, high=100, shape=(4,), dtype=np.float32)
       
        self.info = {}
        self.alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.step_count = 0
        self.antenna = AntennaModel(30)
        self.codebook = self.antenna.gen_codebook(9, 7)
        self.preid = 0
        self.ue_loc = [15,0]
        
    def sweep(self, txpat):
        gains = []
        phi = self.antenna.get_ang(self.ue_loc)
        for codebook_id in txpat:
            phi_s, theta_s = self.codebook[codebook_id]
            g_dBi = self.antenna.calc_ant_gain(theta=90, phi=phi, theta_s=theta_s, phi_s=phi_s)
            gains.append((phi_s, codebook_id, g_dBi))
            max_gain = max(gains, key=lambda x: x[2])
            maxdbi = max_gain[2]
            maxdbiid = max_gain[1]
            self.preid = maxdbiid

            #the following code will return the mechanism number
        if self.ue_loc[1]==0:
            ydif = 0
        else:
            ydif = self.ue_loc[1]-(self.ue_loc[1]-1)

        return maxdbi, maxdbiid, ydif

    #25 beam wide search mechanism
    def localsearch2(self, spec, ptxi=1):
        # todo finish implementation
        # 25 beam wide search
        codebook = np.arange(63)
        codebook.shape = (7,9)
        nrow, ncol = codebook.shape
        i = spec
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        rowtop = rowtopreshape2(row)
        rowbot = rowbotreshape2(row)
        colfront = columnfrontreshape2(column)
        colback = columnbackreshape2(column)
        # basestation beam pattern
        tx = codebook[rowtop:rowbot, colfront:colback]
        tx = tx.flatten()
        tx = np.array(tx)

        return tx

    #9X9 square search mechanism
    def localsearch1(self, spec, ptxi=1):
        # todo finish implementation
        # 9 beam wide search
        codebook = np.arange(63)
        codebook.shape = (7,9)
        nrow, ncol = codebook.shape
        i = spec
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        rowtop = rowtopreshape1(row)
        rowbot = rowbotreshape1(row)
        colfront = columnfrontreshape1(column)
        colback = columnbackreshape1(column)
        tx = codebook[rowtop:rowbot, colfront:colback]
        tx = tx.flatten()
        tx = np.array(tx)
        return tx
    
    def rewardfunction(self, action, spec):
        print('INSIDE ENV FUNCT: ', spec, type(spec))
        rfunc = self.alpha[3] * spec.item() - (1-self.alpha[3]) * self.pp[action]
        return rfunc

    def act(self, action_index):
        if (action_index == 0):
            self.mec=0
            space = self.localsearch1(self.preid)
            sweep = self.sweep(space)
            state = sweep[0], sweep[1], sweep[2], self.mec
        elif (action_index == 1):
            self.mec=1
            space = self.localsearch2(self.preid)
            sweep = self.sweep(space)
            state = sweep[0], sweep[1], sweep[2], self.mec
        elif (action_index == 2):
            self.mec=2
            sweep = self.sweep(self.codebook)
            state = sweep[0], sweep[1], sweep[2], self.mec
        return state
    
    def step(self, action):
        # TODO
        # Step function should go as follows: Action is selection of beam training mechanism hence 4 discrete values as the action space.
        # Each step a beamtraining mechanism is selected and run based on the action provided. A reward is calculated using the reward func below.
        # Each beam training mechanism is assigned a penalty vector as shown by the array self.pp the fist is exhaustive, second is beamsweeptwo and
        # so forth.
        ##########################
        self.state = self.act(action)
        self.spec = self.state[0]
        reward = self.rewardfunction(action, self.spec)
        

        ##########################
    

        self.mec = action

        
        #calculate difference between adjacent distance
        #self.diffdis = bsuedis - self.pevdis
        if self.ue_loc[1] ==6:
                self.ue_loc[1] = 0
        else:
            self.ue_loc[1] = self.ue_loc[1]+1

        self.preid = self.state[1]
        #calculate difference between adjacent angles

        #saving previous angle and distance value to calculate difference for context
        #P return self.state, reward, True, {}
        return self.state, reward, True, {} 


    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = list(self.sweep(self.codebook.keys()))
        self.state.append(2)
        self.spec = self.state[0]
        return self.state
