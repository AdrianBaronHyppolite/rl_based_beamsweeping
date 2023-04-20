from operator import truediv
from gym import Env
from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2
from gym.spaces import Discrete, Box
from toolbox import dB2lin, dBm2watts, watts2dBm
from antenna import AntennaArray
import pandas as pd
import simulation
import fortress
import math
import numpy as np
import random
import gym
import time
# import pack2dict


class RLIAEnv(Env):
    def __init__(self):
        # The RL algorithm should select one of four beam training options
        self.action_space = Discrete(4)
        # observation space
        # Environment values
        self.mec = 0
        self.state = 0
        self.r = watts2dBm(25)
        self.alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.a = -90
        self.A_dB = -30
        self._bs_loc = np.array([0, 0, 0])  # Base station.
        self._ue_loc = np.array([0, 10, 0])  # User equipment.
        self.phiid = -45 #set horizontal point of transmitter
        self.A = dB2lin(self.A_dB)
        self.prevmeasure = 0

        ##########
        self.pevdis = 0
        self.pevang = 0
        self.disfif = 0
        self.angdif = 0
        ##########newspec = 0
        self._Ptx_dBm = 25
        self._fc = 30e9  # Carrier frequency.
        self.element_gain_dbi = 1  # Antenna element gain
        # Wavelength of the 30 GHz carrier frequency.
        self._wavelen = 3e8/self._fc
        self._nx, self._ny, self._nz = 1, 8, 8
        self._M = int(self._nx*self._ny*self._nz)
        self._dx = self._dy = self._dz = self._wavelen * 0.5
        self.nhbeams = 9
        self.nvbeams = 7
        self.hangmin, self.hangmax = -45, 45  # vertical angles
        self.vangmin, self.vangmax = -35, 35  # vertical angles


        self._bs_antenna = AntennaArray(
            self._wavelen, self._nx, self._ny, self._nz, self._dx, self._dy, self._dz, hbeams=self.nhbeams, vbeams=self.nvbeams,
            hangmin=self.hangmin, hangmax=self.hangmax, vangmin=self.vangmin, vangmax=self.vangmax)

        self.nbits = 4


    #execution time tracker

    def calc_spectral_efficiency(self, rsrp, bandwidth):
        """
        Calculates the spectral efficiency given the received signal strength and the available bandwidth.
        The formula used is:
    
        spectral efficiency = bandwidth * log2(1 + 10**(rsrp/10)/bandwidth)
    
        :param rsrp: Received signal strength in dBm.
        :param bandwidth: Available bandwidth in Hz.
        :return: Spectral efficiency in bits/s/Hz.
        """
        rsrp_linear = dBm2watts(rsrp)
        spectral_efficiency = bandwidth * np.log2(1 + rsrp_linear/bandwidth)
        return spectral_efficiency


    def calc_prx(self, Cd, w_bs2ue):
        '''
        This returns the received power in watts.
        '''
        # Don't ask me why this is the conjugate. It just works as in the 
        #   reference MATLAB code.
        psi = np.conj(np.matmul(Cd.T, w_bs2ue))
        # Don't ask me why this is the conjugate. It just works as in the 
        #   reference MATLAB code.
        D=0
        prx = np.power(np.linalg.norm(D+psi), 2)
        return prx


    def sweep(self, rx, _ue_loc, phiid, dis):
        b = random.randint(0, 6)*10
        c = random.randint(0, 6)*10

        _bs_loc = np.array([0, 0, 0])
        d_bs2ue = np.linalg.norm(_bs_loc-_ue_loc)
        # Path loss coefficient.
        A = dB2lin(self.A_dB)
        # Transmit power.
        # Channels.
        # Direct channel, BS -> UE.
        # np.ones((M, 1))#(np.random.randn(M, 1) + 1j*np.random.randn(M, 1))
        self.cd = np.sqrt(A*np.power(d_bs2ue, -
                          dis)/2.)*simulation.gen_fading(self._M, 1)
        sigstrengthvalues = []


        
        start = time.time()
        for codebook_id in self._bs_antenna.codebook_ids[rx]:
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id)
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
    
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi=phiid
                )
            w = w @ w_steer
            w = np.multiply(w, p)
            g = np.power(np.linalg.norm(np.sum(w)), 2) # Gain.
            gain = 10*np.log10(g)
            w_bs2ue = np.multiply(p, g)


            prx = self.calc_prx(self.cd, w_bs2ue)      
            prx_dBm = watts2dBm(prx)
            sigstrengthvalues = np.append(sigstrengthvalues, prx_dBm + 0.001)
        rsrp = max(sigstrengthvalues)
        spectral = self.calc_spectral_efficiency(rsrp, 30)
        tx = sigstrengthvalues
        spec = np.argmax(sigstrengthvalues)
        end = time.time()
        #print("rsrp: ", rsrp)
        #print("spectral: ", spectral)
        #print("        ")

        sweeptime = (end-start)


        #run baseline sweep to determine chosen sweep accuracy
        baselinsigstrength = 0
        for codebook_id in self._bs_antenna.codebook_ids[rx]:
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id)
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
            
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi = phiid
                )
            w = w @ w_steer
            w = np.multiply(w, p)
            prx = np.power(np.linalg.norm(np.sum(w)), 2) # Gain.
            gain = 10*np.log10(prx)
            w_bs2ue = np.multiply(p, prx)

            prx = self.calc_prx(self.cd, w_bs2ue)
            prx_dBm = watts2dBm(prx)
            baselinsigstrength = np.append(baselinsigstrength, prx_dBm + 0.001)
        
        baselinespectral = max(baselinsigstrength)
        differential = baselinespectral - spectral


        return tx, spec, spectral, sweeptime, differential, phiid

    
    #last beam selection search mechanism
    def lastBeam(self, spec, _ue_loc, phiid, dis, ptxi=1):

        codebook = np.zeros(63)
        codebook.shape = (7,9)
        #set location of transmitter and bs
        _bs_loc = np.array([0, 0, 0])
        d_bs2ue = np.linalg.norm(_bs_loc-_ue_loc)
        # Path loss coefficient.
        A = dB2lin(self.A_dB)
        self.cd = np.sqrt(A*np.power(d_bs2ue, -
                          dis)/2.)*simulation.gen_fading(self._M, 1)
        
        sigstrengthvalues = []
        #for codebook_id in self._bs_antenna.codebook_ids[i]:
            # Radiation vector and angle (in ) corresponding to the steering vector.
        start = time.time()


        #####################################3
        start = time.time()
        for codebook_id in self._bs_antenna.codebook_ids[[spec]]:
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id)
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
    
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi=phiid
                )
            w = w @ w_steer
            w = np.multiply(w, p)
            g = np.power(np.linalg.norm(np.sum(w)), 2) # Gain.
            gain = 10*np.log10(g)
            w_bs2ue = np.multiply(p, g)


            prx = self.calc_prx(self.cd, w_bs2ue)      
            prx_dBm = watts2dBm(prx)
            sigstrengthvalues = np.append(sigstrengthvalues, prx_dBm + 0.001)
        
        tx = sigstrengthvalues
        rsrp = max(sigstrengthvalues)
        #print("rsrp: ",rsrp)
        spectral = self.calc_spectral_efficiency(rsrp, 30)
        spec = np.argmax(sigstrengthvalues)
        end = time.time()

        sweeptime = (end-start)


        #run baseline sweep to determine chosen sweep accuracy
        baselinsigstrength = 0
        for codebook_id in self._bs_antenna.codebook_ids:
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id)
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
            
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi = phiid
                )
            w = w @ w_steer 
            w = np.multiply(w, p)
            prx = np.power(np.linalg.norm(np.sum(w)), 2) # Gain.
            gain = 10*np.log10(prx)
            w_bs2ue = np.multiply(p, prx)

            prx = self.calc_prx(self.cd, w_bs2ue)
            prx_dBm = watts2dBm(prx)
            baselinsigstrength = np.append(baselinsigstrength, prx_dBm + 0.001)
        
        baselinespectral = max(baselinsigstrength)
        differential = baselinespectral - spectral


        return tx, spec, spectral, sweeptime, differential, phiid
    
    
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

    # reward function calculation


    


    def rewardfunction(self, action, spec):
        rfunc = self.alpha[2] * spec - (1-self.alpha[2]) * self.pp[action]
        return rfunc

    def act(self, action_index, specid, _ue_loc, dis):
        state = []
        if (action_index == 0):
            state = self.lastBeam(specid, _ue_loc, self.phiid, dis)
            state = state[2], state[3], state[4], state[5]
        if (action_index == 1):
            loc1 = self.localsearch1(specid)
            state = self.sweep(loc1, _ue_loc, self.phiid, dis)
            state = state[2], state[3], state[4], state[5]
        elif (action_index == 2):
            loc2 = self.localsearch2(specid)
            state = self.sweep(loc2, _ue_loc, self.phiid, dis)
            state = state[2], state[3], state[4], state[5]
        elif (action_index == 3):
            state = self.sweep(self._bs_antenna.codebook_ids, _ue_loc, self.phiid, dis)
            state = state[2], state[3], state[4], state[5]
        return state

    def step(self, action):
        # TODO
        # Step function should go as follows: Action is selection of beam training mechanism hence 4 discrete values as the action space.
        # Each step a beamtraining mechanism is selected and run based on the action provided. A reward is calculated using the reward func below.
        # Each beam training mechanism is assigned a penalty vector as shown by the array self.pp the fist is exhaustive, second is beamsweeptwo and
        # so forth.

        a = random.randint(1, 6)*10
        e = random.randint(0, 6)*10
        f = random.randint(0, 6)*10

        self.location= np.array([a, e, f])

        self.mec = action

        #Changes the distance between the transmitter and reciever
        bsuedis = random.uniform(3.5, 15.5)
        
        self.bsuedis = bsuedis

        
        #calculate difference between adjacent distance
        self.diffdis = bsuedis - self.pevdis
    

        premeasure = self.sweep(self._bs_antenna.codebook_ids, self._ue_loc, self.phiid, self.bsuedis)


        self.run = self.act(action, premeasure[1], self._ue_loc, self.bsuedis)
        self.spec = self.run[0]

        self.state = self.spec, action, self.diffdis, self.bsuedis

        reward = self.rewardfunction(action, self.spec)

        #rotates transmitter horizontally
        if self.phiid == 45:
            self.phiid = -45
        else:
            self.phiid = self.phiid+10

        #calculate difference between adjacent angles
        self.diffang = self.phiid - self.pevang

        #saving previous angle and distance value to calculate difference for context
        self.pevdis = bsuedis
        self.pevang = self.phiid

        #P return self.state, reward, True, {}
        return self.state, reward, True, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = 0
        self.spec = 0
        return self.state

