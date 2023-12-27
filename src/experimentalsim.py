from operator import truediv
from gym import Env
from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2
from gym.spaces import Discrete, Box
from antennaOG import AntennaModel
import pandas as pd
import numpy as np
import math
import statistics
from toolbox import dBm2watts, watts2dBm, pathloss, cb, ce, dB2lin, lin2dB, get_angle, listOfTuples, dbmtolin, calc_path_loss_los
from scipy.integrate import dblquad
from IPython.display import display, Math
import random
from sympy import symbols, integrate, sec


 
class RLIAEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.reward = 0
        self.done = False
        self.state = 0
        self.observation_space = Box(low=-100, high=100, shape=(4,), dtype=np.float32)

        self.alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.antenna = AntennaModel(30)
        self.codebook = self.antenna.gen_codebook(9, 7)
        self.preid = 0
        self.ue_loc = [6,0]
        self.bs_loc = [0,0]
        self.ptx_dbm = 25
        self.exclusionzoneX = [1,2]
        self.exclusionzoneY = [5,6]


    def get_dis(self, loc):
        return np.sqrt(loc[0]**2 + loc[1]**2)

    def get_ang(self, loc):
        return np.angle(loc[0] + 1j*loc[1], deg=True)
    
    

    def sweep(self, txpat):
        rssi = []  
        total = 0
        count = 0
        phi = self.antenna.get_ang(self.ue_loc)
        area = 4
        outer_sum = 0
        size = len(txpat)
        for x in self.exclusionzoneX:
                for y in self.exclusionzoneY:
                    evephi = self.antenna.get_ang([x,y])
                    inner_sum = []
                    for codebook_id in txpat:
                        #ue location
                        phi_s, theta_s = self.codebook[codebook_id]
                        g_dBi = self.antenna.calc_ant_gain(theta=90, phi=phi, theta_s=theta_s, phi_s=phi_s)
                        rss = 30 + g_dBi - calc_path_loss_los(30, self.get_dis(self.ue_loc))
                        rssi.append((phi_s, codebook_id, rss))
                        linrss = dbmtolin(rss)
                        snr = 10*np.log(linrss/dbmtolin(-30))

                        #eve location
                        evephi_s, evetheta_s = self.codebook[codebook_id]
                        eveg_dBi = self.antenna.calc_ant_gain(theta=90, phi=evephi, theta_s=evetheta_s, phi_s=evephi_s)
                        everss = 30 + eveg_dBi - calc_path_loss_los(30, self.get_dis([x,y]))
                        linerss = dbmtolin(everss)
                        count += 1
                        inner_sum.append(max(0,cb(linrss)-ce(linerss)))
                        evesnr = 10*np.log(linerss/dbmtolin(-30))
                    outer_sum = np.mean(inner_sum)
                    #print(inner_sum)
        csec = outer_sum/area                            

        maxdbiid = max(rssi, key=lambda x: x[2])
        maxdbi = maxdbiid[2]
        # casting my preid to an int
        self.preid = 31
        #int(maxdbiid[1])

        return maxdbi, self.preid, csec


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
        rfunc = (self.alpha[3] * self.secrecy) - ((1-self.alpha[3]) * self.pp[action])
        return rfunc

    def act(self, action_index):
        state = [0,0,0,0,0,0,0]
        if (action_index == 0):
            self.mec=0
            space = self.localsearch1(self.preid)
            rss, preid, csec = self.sweep(space)
            distance = self.get_dis(self.ue_loc)
            state = rss, preid, self.mec, self.ue_loc[1], distance, csec
        elif (action_index == 1):
            self.mec=1
            space = self.localsearch2(self.preid)
            rss, preid, csec = self.sweep(space)
            distance = self.get_dis(self.ue_loc)
            state = rss, preid, self.mec, self.ue_loc[1], distance, csec
        elif (action_index == 2):
            self.mec=2
            rss, preid, csec = self.sweep(self.codebook)
            distance = self.get_dis(self.ue_loc)
            state = rss, preid, self.mec, self.ue_loc[1], distance, csec
        return state
    
    def step(self, action):
        maxrss, previd, sweepnum, ue_location, ovedistance, cbsp = self.act(action)
        self.state = maxrss, previd, sweepnum, ue_location, ovedistance, cbsp, self.exclusionzoneX, self.exclusionzoneY
        self.secrecy = cbsp
        reward = self.rewardfunction(action)
        

        ##########################
        self.mec = action
        return self.state, reward, True, {} 
    
    
    def teststep(self, action):
         #The following code will call your act function and increase the timestep increment by one each time the step function is called, once the timestep = 12 the UE's y location will be increased by 1 until it reaches 6 then it will be set back to 0 and repeat.
        self.ue_loc[1] += 1
        if self.ue_loc[1] == 7:
            self.ue_loc[1] = 0

        maxrss, previd, sweepnum, ue_location, ovedistance, cbsp = self.act(action)
        self.state = maxrss, previd, sweepnum, ue_location, ovedistance, cbsp, self.exclusionzoneX, self.exclusionzoneY 
        self.secrecy = cbsp
        reward = self.rewardfunction(action)
        ##########################
        self.mec = action
        return self.state, reward, True, {} 
    
    def get_state(self):
        return self.state


    def reset(self):
        # Reset the state of the environment to an initial state
        maxrss, previd, sweepnum, ue_location, ovedistance, cbsp = self.act(2)
        self.state = maxrss, previd, sweepnum, ue_location, ovedistance, cbsp, self.exclusionzoneX, self.exclusionzoneY
        self.spec = cbsp
        return self.state
