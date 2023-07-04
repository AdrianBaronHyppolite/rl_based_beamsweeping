from operator import truediv
from gym import Env
from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2
from gym.spaces import Discrete, Box
from antennaOG import AntennaModel
import pandas as pd
import numpy as np
from toolbox import rss, dBm2watts, watts2dBm, pathloss
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
        self.bs_loc = [0,0]
        self.A_db = -30
        self.ptx_dbm = 25


    def calc_prx(self,codebook_id, p):
        phi_s, theta_s = self.codebook[codebook_id]
        g_dBi = self.antenna.calc_ant_gain(theta=90, phi=0, theta_s=theta_s, phi_s=phi_s)
        return p+g_dBi-pathloss(self.get_dis(self.ue_loc))

    def get_dis(self, loc):
        return np.sqrt(loc[0]**2 + loc[1]**2)

    def get_ang(self, loc):
        return np.angle(loc[0] + 1j*loc[1], deg=True)
        


    def sweep(self, txpat):
        rssi = []
        total = 0
        count = 0
        phi = self.antenna.get_ang(self.ue_loc)
        for codebook_id in txpat:
            phi_s, theta_s = self.codebook[codebook_id]
            g_dBi = self.antenna.calc_ant_gain(theta=90, phi=phi, theta_s=theta_s, phi_s=phi_s)
            rss = self.ptx_dbm+g_dBi-pathloss(self.get_dis(self.ue_loc))
            rssi.append((phi_s, codebook_id, rss))
            count += 1


        maxdbiid = max(rssi, key=lambda x: x[2])
        maxdbi = maxdbiid[2]
        # casting my preid to an int
        self.preid = int(maxdbiid[1])

        return maxdbi, self.preid
    


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
    
    def rewardfunction(self, action):
        rfunc = (self.alpha[3] * self.spec) - ((1-self.alpha[3]) * self.pp[action])
        return rfunc

    def act(self, action_index):
        state = [0,0,0,0,0,0,0]
        if (action_index == 0):
            self.mec=0
            space = self.localsearch1(self.preid)
            sweep = self.sweep(space)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance
        elif (action_index == 1):
            self.mec=1
            space = self.localsearch2(self.preid)
            sweep = self.sweep(space)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance
        elif (action_index == 2):
            self.mec=2
            sweep = self.sweep(self.codebook)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance
        return state
    
    def step(self, action):
        #the following code increments the ue location by 1 each time the step function is called
        self.step_count += 1

        self.ue_loc[1] += 1
        if self.ue_loc[1] == 7:
            self.ue_loc[1] = -1

        actset = self.act(action)
        self.state = actset[0], actset[2], actset[4], actset[5]
        self.spec = self.state[0]
        reward = self.rewardfunction(action)
        

        ##########################
        self.mec = action
        return self.state, reward, True, {} 
    
    
    def teststep(self, action):
         #The following code will call your act function and increase the timestep increment by one each time the step function is called, once the timestep = 12 the UE's y location will be increased by 1 until it reaches 6 then it will be set back to 0 and repeat.
        self.ue_loc[1] += 1
        if self.ue_loc[1] == 7:
            self.ue_loc[1] = 0

        actset = self.act(action)
        self.state = actset[0], actset[2], actset[4], actset[5]
        self.spec = self.state[0]
        reward = self.rewardfunction(action)
        ##########################
        self.mec = action
        return self.state, reward, True, {} 
    
    def get_state(self):
        return self.state


    def reset(self):
        # Reset the state of the environment to an initial state
        actset = self.act(2)
        self.state = actset[0], actset[2], actset[4], actset[5]
        self.spec = self.state[0]
        return self.state
