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
# import pack2dict


class RLIAEnv(Env):
    def __init__(self):
        # The RL algorithm should select one of four beam training options
        self.action_space = Discrete(4)
        # observation space

        # Environment values
        self.state = 0
        self.codebooksize = 63  # adjustable varibale for codebook size
        self.snr = 0.36
        self.r = watts2dBm(25)
        self.alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.a = -90
        self.sig = self.snr/(self.r-self.a)
        self.A_dB = -30
        self._bs_loc = np.array([0, 0, 0])  # Base station.
        self._ue_loc = np.array([50, 0, 0])  # User equipment.
        ##########
        self.moved_bs_loc = np.array([0, 0, 0])  # Base station.
        self.moved_ue_loc = np.array([50, 0, 0])  # User equipment.
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

        self._bs_antenna = AntennaArray(
            self._wavelen, self._nx, self._ny, self._nz, self._dx, self._dy, self._dz, hbeams=self.nhbeams, vbeams=self.nvbeams,
            hangmin=self.hangmin, hangmax=self.hangmax, vangmin=self.vangmin, vangmax=self.vangmax)

        self._bs_antenna_moved = AntennaArray(
            self._wavelen, self._nx, self._ny, self._nz, self._dx, self._dy, self._dz, hbeams=self.nhbeams, vbeams=self.nvbeams,
            hangmin=self.hangmin, hangmax=self.hangmax, vangmin=self.vangmin, vangmax=self.vangmax)

        self.nbits = 4

        self.alpha_bs2ue = 3.5  # ... between BS and user equipment.

        self.d_bs2ue = np.linalg.norm(self._bs_loc-self._ue_loc)
        # Path loss coefficient.
        self.A = dB2lin(self.A_dB)
        # Transmit power.
        # Transmit power per element.
        self.p = np.sqrt(dBm2watts(self._Ptx_dBm)/self._M) * \
            np.ones((self._M, 1))
        # Channels.
        # Direct channel, BS -> UE.
        # np.ones((M, 1))#(np.random.randn(M, 1) + 1j*np.random.randn(M, 1))
        self.cd = np.sqrt(self.A*np.power(self.d_bs2ue, -
                          self.alpha_bs2ue)/2.)*simulation.gen_fading(self._M, 1)

        

        sigstrengthvalues = []
        for codebook_id in self._bs_antenna.codebook_ids:
            # Radiation vector and angle (in ) corresponding to the steering vector.
            w_steer, angle_steer = self._bs_antenna.steering_vec(
                codebook_id, ptxi=1)
            # Radiated power in the direction of interest.
            psi = np.conj(self.cd.T @ w_steer)
            prx = np.power(np.linalg.norm(psi), 2)
            prx_dBm = watts2dBm(prx)

            sigstrengthvalues = np.append(sigstrengthvalues, prx_dBm)

        self.spec = sigstrengthvalues.max()

        self.matrix2d = np.reshape(sigstrengthvalues, (7, 9))

        Movedsigstrengthvalues = []

        for codebook_id in self._bs_antenna_moved.codebook_ids:
            # Radiation vector and angle (in ) corresponding to the steering vector.
            w_steer, angle_steer = self._bs_antenna_moved.steering_vec(
                codebook_id, ptxi=1)
            # Radiated power in the direction of interest.
            Movedpsi = np.conj(self.cd.T @ w_steer)
            Movedprx = np.power(np.linalg.norm(Movedpsi), 2)
            Movedprx_dBm = watts2dBm(Movedprx)
            Movedsigstrengthvalues = np.append(
                Movedsigstrengthvalues, Movedprx_dBm)

        self.MovedNewpatternMatrix = np.reshape(Movedsigstrengthvalues, (7, 9))
        self.Movedspec = Movedsigstrengthvalues.max()
    
    def movedcd(self, bsloc, ueloc):
            d_bs2ue = np.linalg.norm(bsloc-ueloc)
            cd = np.sqrt(self.A*np.power(d_bs2ue, -
                          self.alpha_bs2ue)/2.)*simulation.gen_fading(self._M, 1)
            return cd
    def move(self):
        a = random.randint(0, 6)*10
        b = random.randint(0, 6)*10
        c = random.randint(0, 6)*10
        d = random.randint(0, 6)*10
        e = random.randint(0, 6)*10
        f = random.randint(0, 6)*10

        self.moved_bs_loc = np.array([0, 0, 0])
        self.moved_ue_loc = np.array([d, e, f])
        Movedsigstrengthvalues = []

        cd = self.movedcd(self.moved_bs_loc, self.moved_ue_loc)

        for codebook_id in self._bs_antenna_moved.codebook_ids:
            # Radiation vector and angle (in ) corresponding to the steering vector.
            w_steer, angle_steer = self._bs_antenna_moved.steering_vec(
                codebook_id, ptxi=1)
            # Radiated power in the direction of interest.
            Movedpsi = np.conj(cd.T @ w_steer)
            Movedprx = np.power(np.linalg.norm(Movedpsi), 2)
            Movedprx_dBm = watts2dBm(Movedprx)
            Movedsigstrengthvalues = np.append(
                Movedsigstrengthvalues, Movedprx_dBm)

        self.MovedNewpatternMatrix = np.reshape(Movedsigstrengthvalues, (7, 9))
        self.Movedspec = Movedsigstrengthvalues.max()

    def exhaustive(self, matrix, ptxi=1):
        """
        Inputs:
        - ptxi (dBm): The radiated power that would be radiated if the antenna was isotropic.
        """
        # Beam sweeping.\
        spec = matrix.max()

        tx = matrix
        rx = matrix
        return rx, tx, spec

    def lastBeam(self, iamatrix, movedmatrix, ptxi=1):

        nrow, ncol = movedmatrix.shape
        i = np.argmax(iamatrix)
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        # subset of base station beams
        tx = movedmatrix[row:row+1, column:column+1]
        # subset of transmitter beams
        rx = movedmatrix[row:row+1, column:column+1]
        spec = rx.max()
        return rx, tx, spec

    def squareTwentyFive(self, iamatrix, movedmatrix, ptxi=1):
        # todo finish implementation
        # 25 beam wide search

        nrow, ncol = movedmatrix.shape
        i = np.argmax(iamatrix)
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
        tx = movedmatrix[rowtop:rowbot, colfront:colback]

        rowtoptransmit = rowtopreshape1(row)
        rowbottransmit = rowbotreshape1(row)
        colfronttransmit = columnfrontreshape1(column)
        colbacktransmit = columnbackreshape1(column)
        # transmitter beam pattern
        rx = movedmatrix[rowtoptransmit:rowbottransmit,
                         colfronttransmit:colbacktransmit]
        spec = rx.max()
        return rx, tx, spec

    def squareNine(self, iamatrix, movedmatrix, ptxi=1):
        # todo finish implementation
        # 9 beam wide search

        nrow, ncol = movedmatrix.shape
        i = np.argmax(iamatrix)
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        rowtop = rowtopreshape1(row)
        rowbot = rowbotreshape1(row)
        colfront = columnfrontreshape1(column)
        colback = columnbackreshape1(column)
        rx = movedmatrix[rowtop:rowbot, colfront:colback]
        tx = movedmatrix[rowtop:rowbot, colfront:colback]
        spec = rx.max()
        return rx, tx, spec

    # reward function calculation
    def rewardfunction(self, action, spec):
        rfunc = self.alpha[1] * spec - (1-self.alpha[1]) * self.pp[action]
        return rfunc

    def act(self, action_index):
        if (action_index == 0):
            set = self.lastBeam(self.matrix2d, self.MovedNewpatternMatrix)
        elif (action_index == 1):
            set = self.squareNine(self.matrix2d, self.MovedNewpatternMatrix)
        elif (action_index == 2):
            set = self.squareTwentyFive(
                self.matrix2d, self.MovedNewpatternMatrix)
        elif (action_index == 3):
            set = self.exhaustive(self.MovedNewpatternMatrix)
        return set

    # def spectral_efficency(self, action_index):
    #     a = self.act(action_index)
    #     if (action_index == 0):
    #          spectral = a[2]
    #     elif (action_index == 1):
    #         spectral = a[2]
    #     elif (action_index == 2):
    #         spectral = a[2]
    #     elif (action_index == 3):
    #         spectral = a[2]
    #     return spectral

    def step(self, action, move=False):
        # TODO
        # Step function should go as follows: Action is selection of beam training mechanism hence 4 discrete values as the action space.
        # Each step a beamtraining mechanism is selected and run based on the action provided. A reward is calculated using the reward func below.
        # Each beam training mechanism is assigned a penalty vector as shown by the array self.pp the fist is exhaustive, second is beamsweeptwo and
        # so forth.

        self.state = self.act(action)
        self.spec = self.state[2]

        reward = self.rewardfunction(action, self.spec)

        if(move):
            self.move()

        return self.state, reward, True, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = 0
        self.spec = 0
        return self.state


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