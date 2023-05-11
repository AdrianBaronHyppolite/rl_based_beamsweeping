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
        self.phiid = -5 #set horizontal point of transmitter
        self.A = dB2lin(self.A_dB)
        self.prevmeasure = 0
        self.mec = 0

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

    def wavelength_to_bandwidth(self, wavelength, spectral_width):
        """
        Calculates the bandwidth (in Hertz) for a given wavelength (in meters) and spectral width (in nanometers).

        :param wavelength: Wavelength in meters.
        :param spectral_width: Spectral width in nanometers.
        :return: Bandwidth in Hertz.
        """
        speed_of_light = 299792458  # m/s
        central_frequency = speed_of_light / wavelength
        delta_frequency = central_frequency * (spectral_width / wavelength) * 1e-9
        bandwidth = delta_frequency * 2  # Double-sided bandwidth
        return bandwidth


    def calculate_spectral_efficiency(self, dbm_values, bandwidth):
        """
        Calculates the spectral efficiency from a list of dBm values and bandwidth in Hz.

        :param dbm_values: List of dBm values.
        :param bandwidth: Bandwidth in Hz.
        :return: Spectral efficiency in bits per second per Hz.
        """
        # Convert dBm values to linear scale (milliwatts)
        milliwatts = [10**(dbm/10) for dbm in dbm_values]

        # Sum the milliwatts and divide by the bandwidth to get the watts per Hz
        watts_per_hz = sum(milliwatts) / bandwidth

        # Convert watts per Hz to dBm per Hz
        dbm_per_hz = 10 * math.log10(watts_per_hz)

        # Calculate the spectral efficiency in bits per second per Hz
        spectral_efficiency = math.log2(1 + 10**(dbm_per_hz/10))

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
        d_bs2ue = np.linalg.norm(self._bs_loc-_ue_loc)
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
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id, ptxi=1)
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
    
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi=self.phiid
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
        bandwith = self.wavelength_to_bandwidth(self._wavelen, 0.1)
        spectral = self.calculate_spectral_efficiency(sigstrengthvalues, bandwith)
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
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id, phi=np.deg2rad(phiid))
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
            
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi = 0
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
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id, ptxi=1)
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
    
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi=self.phiid
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
        bandwith = self.wavelength_to_bandwidth(self._wavelen, 0.1)
        spectral = self.calculate_spectral_efficiency(sigstrengthvalues, bandwith)
        #print("rsrp: ",rsrp)
        spec = np.argmax(sigstrengthvalues)
        end = time.time()

        sweeptime = (end-start)


        #run baseline sweep to determine chosen sweep accuracy
        baselinsigstrength = 0
        for codebook_id in self._bs_antenna.codebook_ids:
            w_steer, steering_angle = self._bs_antenna.steering_vec(codebook_id, phi=np.deg2rad(phiid))
            #print(steering_angle, codebook_id)
            w_steer = w_steer.conj()
            
            element_gain = 10*np.power(10, self.element_gain_dbi/10) # dBi -> linear.
            m = int(self._nx*self._ny*self._nz) # Number of antenna elements.
            p = np.sqrt(dBm2watts(self._Ptx_dBm)/m) * np.ones((m, 1)) * element_gain # Power spread over antenna elements.
            w, _ = self._bs_antenna.calc_array_factor( # Radiation pattern.
                    theta=np.pi/2, phi = 0
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
            self.mec=0
            state = self.lastBeam(specid, _ue_loc, self.phiid, dis)
            state = state[2], self.mec, state[3], state[5], state[1]
        if (action_index == 1):
            self.mec=1
            loc1 = self.localsearch1(specid)
            state = self.sweep(loc1, _ue_loc, self.phiid, dis)
            state = state[2], self.mec, state[3], state[5], state[1]
        elif (action_index == 2):
            self.mec=2
            loc2 = self.localsearch2(specid)
            state = self.sweep(loc2, _ue_loc, self.phiid, dis)
            state = state[2], self.mec, state[3], state[5], state[1]
        elif (action_index == 3):
            self.mec=3
            state = self.sweep(self._bs_antenna.codebook_ids, _ue_loc, self.phiid, dis)
            state = state[2], self.mec, state[3], state[5], state[1]
        return state

    def step(self, action, id, bsuedis, location):
        # TODO
        # Step function should go as follows: Action is selection of beam training mechanism hence 4 discrete values as the action space.
        # Each step a beamtraining mechanism is selected and run based on the action provided. A reward is calculated using the reward func below.
        # Each beam training mechanism is assigned a penalty vector as shown by the array self.pp the fist is exhaustive, second is beamsweeptwo and
        # so forth.
        ##########################

        self.state = self.act(action, id, location, bsuedis)
        reward = self.rewardfunction(action, self.state[0])
        ##########################

        self.mec = action

        
        #calculate difference between adjacent distance
        self.diffdis = bsuedis - self.pevdis
    

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

