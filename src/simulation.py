import numpy as np
from toolbox import watts2dBm, dBm2watts, dB2lin, get_angle
import optimizers


def calc_prx(Cd, Cr, G, w_bs2ue, w_bs2ris, Gamma, Theta):
    '''
    This returns the received power in watts.
    '''
    # Don't ask me why this is the conjugate. It just works as in the 
    #   reference MATLAB code.
    psi = np.conj(np.matmul(Cd.T, w_bs2ue))
    # Don't ask me why this is the conjugate. It just works as in the 
    #   reference MATLAB code.
    A = np.conj(np.matmul(Cr.T, Gamma))
    B = np.matmul(A, Theta)
    C = np.matmul(B, G)
    D = np.matmul(C, w_bs2ris)
    prx = np.power(np.linalg.norm(D + psi), 2)
    return prx


def gen_fading(nr, nc):
    return np.random.randn(nr, nc) + 1j*np.random.randn(nr, nc)


def simulate(Ptx_dBm, bs_loc, ue_loc, ris_loc, bs_antenna, M, N, alpha_bs2ue, 
        alpha_bs2ris, alpha_ris2ue, ris_optimizer, estimation_err_dB, 
        nbits, A_dB):
    '''
    Ptx_dBm: Base station's  (BS) transmit power in dBm;
    bs_loc: Base station's location, (x, y, z);
    ue_loc: User equipment's location, (x, y, z);
    ris_loc: RIS' location, (x, y, z);
    bs_antenna: Base station's antenna array;
    M: Number of antenna elements in BS' antenna array;
    N: Number of reflecting elements in RIS;
    alpha_bs2ue: Path loss coefficient between BS and user equipment (UE);
    alpha_bs2ris: Path loss coefficient between BS and RIS;
    alpha_bs2ris: Path loss coefficient between RIS and UE;
    ris_optimizer: Method to optimize the RIS' parameters (reflection & phase);
    estimation_err_dB: MSE of the channel's estimates by RIS in dB;
    b: Number of resolution bits to define the set of discrete phase shifts 
        supported by RIS.
    '''
    Ptx = dBm2watts(Ptx_dBm)
    A = dB2lin(A_dB)
    
    # Distances between base station, RIS, and user equipment.
    d_bs2ris = np.linalg.norm(bs_loc-ris_loc)
    d_ris2ue = np.linalg.norm(ris_loc-ue_loc)
    d_bs2ue  = np.linalg.norm(bs_loc-ue_loc)

    # Channels with small-scale fading (standard normal distribution).
    # Channel error.
    mse = 10.**(estimation_err_dB/10.) # MSE of channel estimation.
    tau = 2.*mse - mse**2.
    # BS-UE, channel and channel estimate by RIS.
    attenuation = np.sqrt(A*np.power(d_bs2ue, -alpha_bs2ue)/2.)
    Cd = attenuation*gen_fading(M, 1)
    Cd_est = np.sqrt(1-tau)*Cd + np.sqrt(tau)*attenuation*gen_fading(M, 1)
    # RIS-UE, channel and channel estimate by RIS.
    attenuation = np.sqrt(A*np.power(d_ris2ue, -alpha_ris2ue)/2.)
    Cr = attenuation*gen_fading(N, 1)
    Cr_est = np.sqrt(1-tau)*Cr + np.sqrt(tau)*attenuation*gen_fading(N, 1)
    # BS-RIS, channel and channel estimate by RIS.
    attenuation = np.sqrt(A*np.power(d_bs2ris, -alpha_bs2ris)/2.)
    G = attenuation*gen_fading(N, M)
    G_est = np.sqrt(1-tau)*G + np.sqrt(tau)*attenuation*gen_fading(N, M)

    # Transmit power.
    p = np.sqrt(Ptx/M) * np.ones((M, 1)) # Transmit power per element.
    # NB Assumptions: 
    #   (a) The BS's antenna is directional;
    #   (b) The RIS does not affect the radiated power apart from
    #       degrading the signal and changing phases;
    #   (c) The UE is equipped with an omnidirectional antenna.
    # Antenna gain, base station -> user equipment.
    phi = 0 # Perfect beam-alignment to user equipment.
    g = bs_antenna.calc_array_factor(theta=np.pi/2, phi=phi).reshape(M, 1)
    w_bs2ue = np.multiply(p, g) # Radiated pattern.
    # Antenna gain, base station -> RIS.
    phi = get_angle(ris_loc-bs_loc, ue_loc-bs_loc)
    g = bs_antenna.calc_array_factor(theta=np.pi/2, phi=phi).reshape(M, 1)
    w_bs2ris = np.multiply(p, g) # Radiated pattern.

    # Received power.
    Gamma, Theta = optimizers.optimize(
        PdBm=Ptx_dBm, 
        w_bs2ue=w_bs2ue, 
        w_bs2ris=w_bs2ris, 
        M=M, 
        N=N, 
        Cd=Cd_est, 
        Cr=Cr_est, 
        G=G_est, 
        optimizer=ris_optimizer,
        nbits=nbits,
    )
    # Sanity-check: To make sure the results make sense. 
    #   a) The values for Gamma should be in [0, 1];
    #   b) The angle values for Theta should be in [0, 2*pi].
    assert np.sum(Gamma > 1) == 0 and np.sum(Gamma < 0) == 0, \
        "Gamma values should not be greater than 1!"
    assert np.sum(np.abs(np.angle(Theta)) > np.pi) == 0, \
        "Theta angles should no be greater than 2pi!"
    # print("Gamma = {}.".format(np.sum(np.abs(Gamma) > 1)))
    prx = calc_prx(Cd, Cr, G, w_bs2ue, w_bs2ris, Gamma, Theta)
    return watts2dBm(prx)