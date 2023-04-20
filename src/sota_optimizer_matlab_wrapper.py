import numpy as np
import matlab.engine 
eng = matlab.engine.start_matlab()

def optimize(PdBm, w_bs2ue, w_bs2ris, M, N, Cd, Cr, G, nbits):
    '''
    State-of-the-art optimizer from paper: 
        Lyu, Bin, Dinh Thai Hoang, Shimin Gong, Dusit Niyato, and Dong In Kim. 
        "IRS-based wireless jamming attacks: When jammers can attack without power.
        " IEEE Wireless Communications Letters 9, no. 10 (2020): 1663-1667.
    '''
    # To MATLAB types.
    _PdBm = matlab.double([PdBm])
    _w_bs2ue = matlab.double(w_bs2ue.tolist(), is_complex=True)
    _w_bs2ris = matlab.double(w_bs2ris.tolist(), is_complex=True)
    _M  = matlab.double([M])
    _N  = matlab.double([N])
    _Cd = matlab.double(Cd.tolist(), is_complex=True)
    _Cr = matlab.double(Cr.tolist(), is_complex=True)
    _G  = matlab.double(G.tolist(), is_complex=True)
    _b  = matlab.double([nbits])
    # Get optimized reflection and phase shift coefficientes.
    [Gamma_dis, Theta_dis] = eng.sota_optimizer(
        _PdBm, _w_bs2ue, _w_bs2ris, _M, _N, _Cd, _Cr, _G, _b, nargout=2
    )
    # From MATLAB to numpy types.
    Gamma_dis = np.diag(np.asanyarray(Gamma_dis).flatten())
    Theta_dis = np.diag(np.asanyarray(Theta_dis).flatten())
    # Gamma_con = np.diag(np.asanyarray(Gamma_con).flatten())
    # Theta_con = np.diag(np.asanyarray(Theta_con).flatten())
    # return Gamma_dis, Theta_dis, Gamma_con, Theta_con
    return Gamma_dis, Theta_dis