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
import gym
# import pack2dict


class RLIAEnv(Env):
    def __init__(self):
        # The RL algorithm should select one of four beam training options
        self.action_space = Discrete(4)
        # observation space
        self.observation_space = Box(low=-50, high=-7, shape=(7,9), dtype=np.float16)
        # Environment values
        self.state = 0
        self.alpha = 0.5
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.a = -90
        self.sig = self.snr/(self.r-self.a)
        self.A_dB = -30
        ##########
        ##########newspec = 0

    
    #exhaustive search mechanism
    def exhaustive(self, matrix):
        return matrix
    

    #last beam selection search mechanism
    def lastBeam(self, premeasure, movedmatrix, ptxi=1):
        nrow, ncol = movedmatrix.shape
        i = np.argmax(premeasure)
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        # subset of base station beams
        transmitpattern = movedmatrix[row:row+1, column:column+1]
        spec = transmitpattern.max()
        return transmitpattern
    
    
    #25 beam wide search mechanism
        #last beam selection search mechanism
    def lastBeam(self, premeasure, movedmatrix, ptxi=1):

        nrow, ncol = movedmatrix.shape
        i = np.argmax(premeasure)
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        # subset of base station beams
        tx = movedmatrix[row:row+1, column:column+1]
        # subset of reciever beams
        rx = movedmatrix[row:row+1, column:column+1]
        spec = rx.max()
        return rx, tx, spec
    
    
    #25 beam wide search mechanism
    def squareTwentyFive(self, premeasure, movedmatrix, ptxi=1):
        nrow, ncol = movedmatrix.shape
        i = np.argmax(premeasure)
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
        # reciever beam pattern
        rx = movedmatrix[rowtoptransmit:rowbottransmit,
                         colfronttransmit:colbacktransmit]
        spec = rx.max()
        return rx, self.moved_ue_loc[0], spec
    

    #9X9 square search mechanism
    def squareNine(self, premeasure, movedmatrix, ptxi=1):
        nrow, ncol = movedmatrix.shape
        i = np.argmax(premeasure)
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
        return rx, self.moved_ue_loc[0], spec

    # reward function calculation


    


    def rewardfunction(self, action, spec):
        rfunc = self.alpha * spec - (1-self.alpha) * self.pp[action]
        return rfunc

    def act(self, action_index):
        if (action_index == 0):
            set = self.lastBeam(self.rxmatrix, self.MovedNewpatternMatrix)
        elif (action_index == 1):
            set = self.squareNine(self.rxmatrix, self.MovedNewpatternMatrix)
        elif (action_index == 2):
            set = self.squareTwentyFive(self.rxmatrix, self.MovedNewpatternMatrix)
        elif (action_index == 3):
            set = self.exhaustive(self.MovedNewpatternMatrix)
        return set

    def step(self, action, move=False):

        self.state = self.act(action)
        self.spec = self.state[2]
        self.rxmatrix = self.state[0]
        self.txmatrix = self.state[1]

        reward = self.rewardfunction(action, self.spec)

        if(move):
            self.move()

        return self.state, reward, True, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = 0
        self.spec = 0
        return self.state

