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
        self.action_space = Discrete(4)
        self.reward = 0
        self.done = False
        self.state = 0
        self.observation_space = Box(low=-100, high=100, shape=(4,), dtype=np.float32)

        self.info = {}
        self.step_count = 0
        self.alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.antenna = AntennaModel(30)
        self.codebook = self.antenna.gen_codebook(9, 7)
        self.preid = 0
        self.ue_loc = [15,0]
        self.bs_loc = [0,0]
        self.ptx_dbm = 30
        self.exclu_locx = [1,4]
        self.exclu_locy = [5,8]


    def get_dis(self, loc):
        return np.sqrt(loc[0]**2 + loc[1]**2)

    def get_ang(self, loc):
        return np.angle(loc[0] + 1j*loc[1], deg=True)
    

    def calc_path_loss_los(self,fc_ghz, d_2d_meter):
        H_UT  = 1.5 # Height of UT (m).
        H_BS  = 10 # Height of BS (m).
        d_bp = 4*H_BS*H_UT*fc_ghz/0.3 # Breaking point distance.
        d_3d = np.sqrt(np.power(d_2d_meter, 2) + np.power(H_BS-H_UT, 2))
        pl = np.empty_like(d_2d_meter).astype(np.float64)
        # PL1: d_2d <= d_bp
        cond = d_2d_meter <= d_bp
        # pl[cond] = 28 + 22*np.log10(d_3d[cond]) + 20*np.log10(fc_ghz)
        pl[cond] = 32.4 + 21*np.log10(d_3d[cond]) + 20*np.log10(fc_ghz)
        # PL2: d_2d > d_bp
        cond = np.invert(cond)
        # pl[cond] = 28 + 40*np.log10(d_3d[cond]) + 20*np.log10(fc_ghz) \
        #     - 9*np.log10(np.power(d_bp, 2)+np.power(H_BS-H_UT, 2))
        pl[cond] = 32.4 + 40*np.log10(d_3d[cond]) + 20*np.log10(fc_ghz) \
            - 9.5*np.log10(np.power(d_bp, 2)+np.power(H_BS-H_UT, 2))
        return pl
        
    
    def calc_path_loss_umi(fc_ghz, d_2d_meter, is_los):
        # Constants
        H_UT = 1.5  # Height of UT (m)
        H_BS = 10  # Height of BS (m)
        fc_mhz = fc_ghz * 1000  # Frequency in MHz
        d_3d = np.sqrt(np.power(d_2d_meter, 2) + np.power(H_BS - H_UT, 2))  # 3D distance

        # Path Loss for LoS
        if is_los:
            # 3GPP UMi-Street Canyon LoS Path Loss Model
            pl = 32.4 + 20 * np.log10(d_3d) + 20 * np.log10(fc_mhz)
        else:
            # 3GPP UMi-Street Canyon NLoS Path Loss Model
            pl = 22.4 + 35.3 * np.log10(d_3d) + 21.3 * np.log10(fc_mhz) - 0.3 * (H_UT - 1.5)

        # Add Shadowing (Log-Normal Fading)
        shadowing_std_dev = 4 if is_los else 7.82  # Standard deviation in dB
        shadowing = np.random.normal(0, shadowing_std_dev)
        pl += shadowing

        # Add Rayleigh Fading for NLoS
        if not is_los:
            # Rayleigh fading - magnitude of a complex Gaussian random variable
            real_part = np.random.normal(0, np.sqrt(0.5))
            imag_part = np.random.normal(0, np.sqrt(0.5))
            rayleigh_fading = 20 * np.log10(np.abs(real_part + 1j * imag_part))  # Convert to dB
            pl += rayleigh_fading

        return pl


    def sweep(self, txpat):
        rssi = []  
        total = 0
        count = 0
        phi = self.antenna.get_ang(self.ue_loc)
        xs = self.exclu_locx
        ys = self.exclu_locy
        area = 9
        outer_sum = 0
        size = len(txpat)
        for x in xs:
                for y in ys:
                    evephi = self.antenna.get_ang([x,y])
                    inner_sum = []
                    for codebook_id in txpat:
                        #ue location
                        phi_s, theta_s = self.codebook[codebook_id]
                        g_dBi = self.antenna.calc_ant_gain(theta=90, phi=phi, theta_s=theta_s, phi_s=phi_s)
                        rss = self.ptx_dbm + g_dBi - self.calc_path_loss_los_umi(30, self.get_dis(self.ue_loc), False)
                        rssi.append((phi_s, codebook_id, rss))
                        linrss = dbmtolin(rss)
                        snr = 10*np.log(linrss/dbmtolin(-30))

                        #eve location
                        evephi_s, evetheta_s = self.codebook[codebook_id]
                        eveg_dBi = self.antenna.calc_ant_gain(theta=90, phi=evephi, theta_s=evetheta_s, phi_s=evephi_s)
                        everss = 30 + eveg_dBi - self.calc_path_loss_los(30, self.get_dis([x,y]))
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
        self.preid = 32
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
    
    #7X7 square search mechanism
    def sevenby7(self, spec):
        # Calculate the row and column indices
        codebook = np.arange(63)
        codebook.shape = (7,9)
        nrow, ncol = codebook.shape
        rid = spec//ncol
        cid = spec % ncol

        # Calculate the indices for slicing the codebook to get a 7x7 matrix
        row_start = max(0, rid - 3)
        row_end = min(nrow, rid + 4)
        col_start = max(0, cid - 3)
        col_end = min(ncol, cid + 4)

        # Slice the codebook to get the 7x7 matrix
        tx = codebook[row_start:row_end, col_start:col_end]
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
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2]
        elif (action_index == 1):
            self.mec=1
            space = self.localsearch2(self.preid)
            sweep = self.sweep(space)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2]
        elif (action_index ==2):
            self.mec=2
            space = self.sevenby7(self.preid)
            sweep = self.sweep(space)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2] 

        elif (action_index == 3):
            self.mec=3
            sweep = self.sweep(self.codebook)
            angle = self.get_ang(self.ue_loc)
            distance = self.get_dis(self.ue_loc)
            state = sweep[0], sweep[1], self.mec, self.ue_loc[1], angle, distance, sweep[2]
        return state
    
    def step(self, action):
        #move the the UE y direction +1 for 6 steps then reset to 0 and do repeat
        self.ue_loc[1] += 1
        if self.ue_loc[1] == 7:
            self.ue_loc[1] = 0


        actset = self.act(action)
        self.state = actset[0], actset[2], actset[4]
        self.spec = self.state[0]

        #security reward function
        reward = actset[6]

        #efficiency reward function
        #reward = self.rewardfunction(action)
        self.cbsp = actset[6]
        

        ##########################
        self.mec = action
        return self.state, reward, True, {} 
    
    
    def teststep(self, action):
         #The following code will call your act function and increase the timestep increment by one each time the step function is called, once the timestep = 12 the UE's y location will be increased by 1 until it reaches 6 then it will be set back to 0 and repeat.
        #self.ue_loc[1] += .5
        #if self.ue_loc[1] == 7:
        #    self.ue_loc[1] = 0

        #move the UE 10 degrees each run until it reaches 60 degrees then reset to 0
        self.ue_loc[0] = 15*np.cos(np.deg2rad(self.step_count*10))
        self.ue_loc[1] = 15*np.sin(np.deg2rad(self.step_count*10))
        if self.step_count == 7:
            self.step_count = 1
            self.ue_loc[0] = 15
            self.ue_loc[1] = 0


        actset = self.act(action)
        self.state = actset[0], actset[2], actset[4]
        self.spec = self.state[0]
        reward = actset[6]
        #self.rewardfunction(action)
        self.cbsp = actset[6]
        ##########################
        self.mec = action

        self.step_count += 1
        return self.state, reward, True, {} 
    
    def get_state(self):
        return self.state


    def reset(self):
        # Reset the state of the environment to an initial state
        actset = self.act(2)
        self.state = actset[0], actset[2], actset[4]
        self.spec = self.state[0]
        return self.state
