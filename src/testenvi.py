from operator import truediv
from gym import Env
from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2
from gym.spaces import Discrete, Box
from toolbox import dB2lin, dBm2watts, watts2dBm
from antenna import AntennaArray
import pandas as pd
import simulation
import fortress
import optimizers
import math
import numpy as np
import random
import time
# import pack2dict


class RLIAEnv(Env):
    def __init__(self):
        # The RL algorithm should select one of four beam training options
        self.action_space = Discrete(4)
        # observation space

        # Environment values
        self.locdif = 0
        self.state = []
        self.codebooksize = 63  # adjustable varibale for codebook size
        self.snr = 0.36
        self.r = watts2dBm(25)
        self.alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.a = -90
        self.sig = self.snr/(self.r-self.a)
        self.A_dB = -30
        self._bs_loc = np.array([0, 0, 0])  # Base station.
        self.prev_ue_loc = np.array([0, 50, 0])  # User equipment.
        ##########
        self._Ptx_dBm = 25
        self._fc = 30e9  # Carrier frequency.
        self.element_gain_dbi = 1  # Antenna element gain
        # Wavelength of the 30 GHz carrier frequency.
        self._wavelen = 3e8/self._fc
        self._nx, self._ny, self._nz = 1, 7, 9
        self._M = int(self._nx*self._ny*self._nz)
        self._dx = self._dy = self._dz = self._wavelen * 0.5
        self.nhbeams = 9
        self.nvbeams = 7
        self.hangmin, self.hangmax = -45, 45  # vertical angles
        self.vangmin, self.vangmax = -35, 35  # vertical angles

        self._bs_loc = np.array([0, 0, 0])
        #self._ue_loc = np.array([10, 10, 10])

        self._bs_antenna = AntennaArray(
            self._wavelen, self._nx, self._ny, self._nz, self._dx, self._dy, self._dz, hbeams=self.nhbeams, vbeams=self.nvbeams,
            hangmin=self.hangmin, hangmax=self.hangmax, vangmin=self.vangmin, vangmax=self.vangmax)

        self.nbits = 4

        self.alpha_bs2ue = 3.5  # ... between BS and user equipment.

        # Path loss coefficient.
        self.A = dB2lin(self.A_dB)
        # Transmit power.
        # Transmit power per element.
        self.p = np.sqrt(dBm2watts(self._Ptx_dBm)/self._M) * \
            np.ones((self._M, 1))
        # Channels.
        # Direct channel, BS -> UE.
        # np.ones((M, 1))#(np.random.randn(M, 1) + 1j*np.random.randn(M, 1))

    
    def sweep(self, rx,_ue_loc):
        Movedsigstrengthvalues = []

        d_bs2ue = np.linalg.norm(self._bs_loc-_ue_loc)

        cd = np.sqrt(self.A*np.power(d_bs2ue, -
                          self.alpha_bs2ue)/2.)*simulation.gen_fading(self._M, 1)

        sigstrengthvalues = []
        start = time.time()
        for codebook_id in self._bs_antenna.codebook_ids[rx]:
            # Radiation vector and angle (in ) corresponding to the steering vector.
            w_steer, angle_steer = self._bs_antenna.steering_vec(
                codebook_id, ptxi=1)
            # Radiated power in the direction of interest.
            psi = np.conj(cd.T @ w_steer)
            prx = np.power(np.linalg.norm(psi), 2)
            prx_dBm = watts2dBm(prx)
            sigstrengthvalues = np.append(sigstrengthvalues, prx_dBm)
        spec = np.argmax(sigstrengthvalues)
        rsrp = sigstrengthvalues.max()
        end = time.time()
        executetime = end - start

        #run baseline sweep to determine chosen sweep accuracy
        baselinesigstrength = []
        baselinestart = time.time()
        for codebook_id in self._bs_antenna.codebook_ids:
            # Radiation vector and angle (in ) corresponding to the steering vector.
            w_steer, angle_steer = self._bs_antenna.steering_vec(
                codebook_id, ptxi=1)
            # Radiated power in the direction of interest.
            psi = np.conj(cd.T @ w_steer)
            prx = np.power(np.linalg.norm(psi), 2)
            prx_dBm = watts2dBm(prx)
            baselinesigstrength = np.append(baselinesigstrength, prx_dBm)
        baselinersrp = baselinesigstrength.max()
        baselineend = time.time()
        blexecutiontime = baselineend-baselinestart
        
        differential = baselinersrp - rsrp
        #converting ue location into int
        

        return sigstrengthvalues, spec, rsrp, executetime, differential, baselinersrp, blexecutiontime

    def lastBeam(self, spec,_ue_loc, ptxi=1):

        d_bs2ue = np.linalg.norm(self._bs_loc-_ue_loc)

        cd = np.sqrt(self.A*np.power(d_bs2ue, -
                          self.alpha_bs2ue)/2.)*simulation.gen_fading(self._M, 1)


        sigstrengthvalues = []
        # Radiation vector and angle (in ) corresponding to the steering vector.
        start = time.time()
        w_steer, angle_steer = self._bs_antenna.steering_vec(
        spec, ptxi=1)
        # Radiated power in the direction of interest.
        psi = np.conj(cd.T @ w_steer)
        prx = np.power(np.linalg.norm(psi), 2)
        prx_dBm = watts2dBm(prx)
        sigstrengthvalues = prx_dBm
        end = time.time()
        executetime = end - start

        #run baseline sweep to determine chosen sweep accuracy
        baselinesigstrength = []
        

        blinestart =time.time()
        for codebook_id in self._bs_antenna.codebook_ids:
            # Radiation vector and angle (in ) corresponding to the steering vector.
            w_steer, angle_steer = self._bs_antenna.steering_vec(
                codebook_id, ptxi=1)
            # Radiated power in the direction of interest.
            psi = np.conj(cd.T @ w_steer)
            prx = np.power(np.linalg.norm(psi), 2)
            prx_dBm = watts2dBm(prx)
            baselinesigstrength = np.append(baselinesigstrength, prx_dBm)
        baselinersrp = baselinesigstrength.max()
        baslineend = time.time()
        differential = baselinersrp - sigstrengthvalues
        blexecutiontime = baslineend - blinestart

        return sigstrengthvalues, executetime, differential, baselinersrp, blexecutiontime

    def squareTwentyFive(self, spec, ptxi=1):
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
    def squareNine(self, spec, ptxi=1):
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
        rfunc = self.alpha[3] * spec - (1-self.alpha[3]) * self.pp[action]
        return rfunc

    def act(self, action_index, specid, _ue_loc):
        if (action_index == 0):
            mec = 0
            set = self.lastBeam(specid,_ue_loc)
            set = set[0], set[1], set[2], set[3], set[4], mec
        elif (action_index == 1):
            loc1 = self.squareNine(specid)
            set = self.sweep(loc1, _ue_loc)
            mec = 1
            set = set[2], set[3], set[4], set[5], set[6], mec
        elif (action_index == 2):
            loc2 = self.squareTwentyFive(specid)
            set = self.sweep(loc2, _ue_loc)
            mec = 2
            set = set[2], set[3], set[4], set[5], set[6], mec
        elif (action_index == 3):
            set = self.sweep(self._bs_antenna.codebook_ids, _ue_loc)
            mec = 3
            set = set[2], set[3], set[4], set[5], set[6], mec
        return set

    def step(self, action, move=False):
        # TODO
        # Step function should go as follows: Action is selection of beam training mechanism hence 4 discrete values as the action space.
        # Each step a beamtraining mechanism is selected and run based on the action provided. A reward is calculated using the reward func below.
        # Each beam training mechanism is assigned a penalty vector as shown by the array self.pp the fist is exhaustive, second is beamsweeptwo and
        # so forth.

        a = random.randint(2, 6)*10
        e = random.randint(0, 6)*10
        f = random.randint(0, 6)*10

        self._ue_loc_= np.array([a, e, f])

        premeasure = self.sweep(self._bs_antenna.codebook_ids, self.prev_ue_loc)

        self.locdif = self._ue_loc_[1]-self.prev_ue_loc[1]

        self.state = self.act(action, premeasure[1], self._ue_loc_)
        self.spec = self.state[0]


        reward = self.rewardfunction(action, self.spec)

        self.prev_ue_loc = self._ue_loc_

        return self.state, reward, True, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = []
        self.spec = 0
        # return self.state
        return np.array([0, 0, 0], dtype = float)


# def main():
#     env = RLIAEnv()
#     rewards = 0
#     n_episodes = 10
#     for i in range(n_episodes):
#         action = env.action_space.sample()
#         observation, reward, done, _ = env.step(action)
#         # print(
#         #     f'observation: {observation}, \nreward: {reward}, \ndone: {done}')
#         rewards += reward
#     mean_of_rewards = rewards/n_episodes
#     print(f'mean of rewards = {mean_of_rewards}')

# if __name__ == "__main__":
#     main()