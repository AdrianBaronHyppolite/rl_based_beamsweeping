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
import math
from scipy.integrate import dblquad
from IPython.display import display, Math
from sympy import symbols, integrate, sec


 
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
        self.alphatwo = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.step_count = 0
        self.antenna = AntennaModel(30)
        self.codebook = self.antenna.gen_codebook(9, 7)
        self.preid = 0
        self.ue_loc = [15,0]


    def capacity_link(self, prx):
        cp = 1/2*np.log(1+(prx/-30))
        return cp
    
    def secpres(self, prx, space):
        f = lambda x,y :(1/2)*np.log(1+(prx/(-30+60)))-(1/2)*np.log(1+(15.4389/(-30+60)))
        integral, error = dblquad(f,0, 15, lambda x: 0, lambda x: 15)
        #print(integral)
        secpres = (1/space)* integral
        return secpres


    def get_dis(self, loc):
        return np.sqrt(loc[0]**2 + loc[1]**2)

    def get_ang(self, loc):
        return np.angle(loc[0] + 1j*loc[1], deg=True)
        
    def sweep(self, txpat):
        gains = []
        caplinks = []
        count = 0
        phi = self.antenna.get_ang(self.ue_loc)
        for codebook_id in txpat:
            phi_s, theta_s = self.codebook[codebook_id]
            g_dBi = self.antenna.calc_ant_gain(theta=90, phi=phi, theta_s=theta_s, phi_s=phi_s)
            gains.append((phi_s, codebook_id, g_dBi))
            #The following gathers capacity links at all beams
            cl = self.capacity_link(g_dBi)
            caplinks.append(cl)
            count += 1

        space = count

        maxdbiid = max(gains, key=lambda x: x[2])
        maxdbi = maxdbiid[2]
        psec = self.secpres(maxdbi, space)
        # casting my preid to an int
        self.preid = int(maxdbiid[1])
        #print("this is my new beamid", self.preid)

            #the following code will return the mechanism numbe
            #the following code will return the mechanism number
        return maxdbi, self.preid, psec

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
    
    def rewardfunction(self, action, psec):
        rfunc = (self.alpha[3] * self.spec) - ((1-self.alpha[3]) * self.pp[action]) - (psec * self.alphatwo[1])
        return rfunc

    def act(self, action_index):
        if (action_index == 0):
            self.mec=0
            space = self.localsearch1(self.preid)
            sweep = self.sweep(space)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2]
        elif (action_index == 1):
            self.mec=1
            space = self.localsearch2(self.preid)
            sweep = self.sweep(space)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2]
        elif (action_index == 2):
            self.mec=2
            sweep = self.sweep(self.codebook)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2]
        return state
    
    def step(self, action):
         #The following code will call your act function and increase the timestep increment by one each time the step function is called, once the timestep = 12 the UE's y location will be increased by 1 until it reaches 6 then it will be set back to 0 and repeat.
        self.step_count += 1
        # if self.step_count == 12:
        #     self.ue_loc[1] += 1
        #     self.step_count = 0
        self.ue_loc[1] += 1
        if self.ue_loc[1] == 7:
            self.ue_loc[1] = -1

        actset = self.act(action)
        self.state = actset[0], actset[2], actset[4], actset[5], actset[6]
        self.spec = self.state[0]
        reward = self.rewardfunction(action, self.state[4])
        

        ##########################
        self.mec = action

        #saving previous angle and distance value to calculate difference for context
        #P return self.state, reward, True, {}
        return self.state, reward, True, {} 
    
    def teststep(self, action):
         #The following code will call your act function and increase the timestep increment by one each time the step function is called, once the timestep = 12 the UE's y location will be increased by 1 until it reaches 6 then it will be set back to 0 and repeat.
        self.ue_loc[1] += 1
        if self.ue_loc[1] == 7:
            self.ue_loc[1] = 0

        actset = self.act(action)
        self.state = actset[0], actset[2], actset[4], actset[5], actset[6]
        self.spec = self.state[0]
        reward = self.rewardfunction(action, self.state[4])
        

        ##########################
        self.mec = action

        #saving previous angle and distance value to calculate difference for context
        #P return self.state, reward, True, {}
        return self.state, reward, True, {} 
    
    def get_state(self):
        return self.state


    def reset(self):
        # Reset the state of the environment to an initial state
        actset = self.act(2)
        self.state = actset[0], actset[2], actset[4], actset[5], actset[6]
        self.spec = self.state[0]
        return self.state
