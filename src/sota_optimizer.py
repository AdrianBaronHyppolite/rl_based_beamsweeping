### The original code is from "IRS-based Wireless Jamming Attacks: How Jammers can Attack without Power?"
### Code modified by André Gomes, gomesa@vt.edu.

import numpy as np
import cvxpy as cp
import scipy.io as sio
from scipy.stats import norm


def myrandn(nr, nc):
    return norm.ppf(np.random.rand(nr, nc))


def optimize(PdBm, w_bs2ue, w_bs2ris, M, N, Cd, Cr, G, b):
    np.random.seed(0)

    # Just for the sake of compatibility with former scripts.
    if w_bs2ue.ndim > 1:
        w_bs2ue = w_bs2ue.flatten()
    if w_bs2ris.ndim > 1:
        w_bs2ris = w_bs2ris.flatten()

    # Hyperparameters.
    P = 1e-3 * 10**(PdBm/10) # Transmit power.
    Iteration = 1000 # the number of Gaussian randomization method
    L = 2**b
    thetaStep = 2*np.pi/L
    thetaSet = np.arange(0, 2*np.pi, thetaStep)
    
    ##############################################################
    # Solve P2 with the given amplitude reflection coefficients  #
    ##############################################################
    Gamma = np.eye(N) # the amplitude reflection coefficient
    A = np.matmul(np.matmul(np.diag(np.matmul(Cr.conj().T,Gamma).squeeze()),G),w_bs2ris)
    psi = np.matmul(Cd.conj().T, w_bs2ue)
    #Create auxiliary matrix #[ A * A',  A * psi';  A' * psi, 0]
    R_1 = np.concatenate((A[:,None] @ A.conj()[None,:],(A*psi.conj())[:,None]),axis=1)
    R_2 = np.concatenate((A.conj()[None,:]*psi, [[0]]),axis=1)
    R = np.concatenate((R_1,R_2),axis=0)

    #CPX Semidefinite programming formulation; The auxiliary variable has one extra entry
    V = cp.Variable((N+1,N+1), complex=True)#, symmetric=True)
    constraints = [V >> 0]
    constraints += [V[i,i] == 1 for i in range(N+1)]
    prob2 = cp.Problem(
        cp.Minimize(cp.real(cp.trace(R @ V)) + cp.abs(psi)**2), constraints)
    prob2.solve()

    ##### Compute the SVD of V
    # U1, S, U2 = np.linalg.svd(V.value) # SVD (S is given in vector form)
    U1, S, U2 = np.linalg.svd(V.value)
    S = np.diag(S) #need to convert S to diagonal matrix form
    
    # Gaussian randomization method
    vBarSet = np.zeros([N,Iteration], dtype=np.csingle)
    PrxSet = np.zeros([Iteration])
    np.random.seed(0)
    for ii in np.arange(Iteration):
        # refer to Eq. 3;
        muBar = U1 @ np.sqrt(S) @ (np.sqrt(1./2)*myrandn(N+1, 1) + np.sqrt(1./2)*1j*myrandn(N+1, 1))
        # refer to Eq. 4;
        vtemp = muBar / muBar[N]
        thetaObtained = -np.angle(vtemp[0:N])
        
        # To make sure all values are in [0, 2pi].
        while np.sum(thetaObtained < 0) > 0:
            thetaObtained[thetaObtained < 0] += 2*np.pi 
        while np.sum(thetaObtained > 2*np.pi) > 0:
            thetaObtained[thetaObtained > 2*np.pi] -= 2*np.pi 
        assert np.sum(thetaObtained < 0)==0 and np.sum(thetaObtained > 2*np.pi)==0,\
            "The theta range is out [0, 2pi]."
        # Quantization.
        idxs = np.argmin(np.abs(thetaObtained-thetaSet), axis=1)
        thetaDis = thetaSet[idxs]
        # Dealing with the borders. This way, values close to 2pi are quantized as 0
        #   rather than the max value in the thetaSet.
        thetaDis[(2*np.pi - thetaObtained < thetaStep/2).flatten()] = 0

        # Estimate the received power based on the obtained phase coefficients.
        vBarSet[:,ii] = np.conj(np.exp(1j*thetaDis.squeeze()))
        PrxSet[ii] = np.real(vBarSet[:,ii][None,:].conj() @ R_1[:,:N] @ vBarSet[:,ii] \
                                + vBarSet[:,ii][None,:].conj() @ R_1[:,N] \
                                + psi * A.conj()[None,:] @ vBarSet[:,ii] + np.abs(psi)**2)\

    ##############################################################
    # Solve P3 with the given phase shifts                       #
    ##############################################################
    bestIndex = np.argmin(PrxSet)
    vBarBest = np.conj(vBarSet[:,bestIndex]) # best performing thetaDis
    C = np.diag(Cr.conj().squeeze()) @ np.diag(vBarBest) @ G @ w_bs2ris
    #Convex solver
    betaCVX = cp.Variable((1, N), nonneg=True)
    constraints = [betaCVX - np.ones((N, 1)) <= 0]
    prob3 = cp.Problem(cp.Minimize(cp.norm(betaCVX @ C + psi)), constraints)
    prob3.solve()

    Gamma = betaCVX.value
    Theta = vBarBest 
    return np.diag(Gamma.flatten()), np.diag(Theta.flatten())