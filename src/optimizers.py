import numpy as np
# from sota_optimizer_matlab_wrapper import optimize as optimize_sota_matlab
from sota_optimizer import optimize as optimize_sota_python

def optimize(PdBm, w_bs2ue, w_bs2ris, M, N, Cd, Cr, G, optimizer, nbits):
    '''
    This function returns Gamma and Theta, which map onto the reflection
    coefficients and phase shifts in the RIS.

    PdBm: Transmit power in dBm;
    w_bs2ue: Radiated power from BS to UE;
    w_bs2ris: Radiated power through RIS;
    M: Number of antenna elements of BS;
    N: Number of reflecting elements of RIS;
    Cd: Channel between BS-UE;
    Cr: Channel between RIS-UE;
    G: Channel between BS-RIS;
    optimizer: Method to optimize the RIS reflection and phase shift parameters.
    nbits: Number of bits for phase shift resolution.

    NB It assumes knowledge about the communication channel between,
        (a) BS - RIS,
        (b) RIS - UE, and
        (c) BS - UE,
    as in the reference paper. 
    '''
    if optimizer == "sota matlab": # State-of-the-art MATLAB implementation.
        Gamma_dis, Theta_dis = None, None # Uncomment to revive the MATLAB implementation.
        # Gamma_dis, Theta_dis = \
            # optimize_sota_matlab(PdBm, w_bs2ue, w_bs2ris, M, N, Cd, Cr, G, nbits)
    elif optimizer == "sota python": # State-of-the-art Python implementation.
        Gamma_dis, Theta_dis = \
            optimize_sota_python(PdBm, w_bs2ue, w_bs2ris, M, N, Cd, Cr, G, nbits)
    elif optimizer == "random":
        # Random Gamma and Theta.
        beta = np.random.rand(N)
        Gamma_dis = np.diag(beta)
        # Theta is selected from a discrete set F of resolution n bits (nbits).
        L = 2**nbits
        F = np.arange(0, 2*np.pi, 2*np.pi/L)
        # theta = np.random.rand(N)*2*np.pi
        theta = np.random.choice(F, size=N)
        Theta_dis = np.diag(np.exp(1j*theta))
        theta = np.random.rand(N)*2*np.pi 
    elif optimizer == "none":
        beta = np.zeros(N)
        Gamma_dis = np.diag(beta)
        theta = np.zeros(N)
        # Theta = np.diag(np.exp(1j*theta))
        Theta_dis = np.diag(theta)
    elif optimizer == "fixed ris":
        beta = np.ones(N)
        Gamma_dis = np.diag(beta)
        theta = np.zeros(N)
        Theta_dis = np.diag(np.exp(1j*theta))
    else:
        assert True == False, \
            "Error, invalid optimizer! Either [sota, random, none]."
    return Gamma_dis, Theta_dis