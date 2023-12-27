import numpy as np 



def pack2dict(dictionary, values):
    for key, value in zip(dictionary.keys(), values):
        dictionary[key].append(value)
    return dictionary


def dBm2watts(dBm):
    # return np.power(10., (dBm-30.)/10.)
    return 1e-3 * np.power(10., dBm/10.)


def dBw2watts(dBw):
    return np.power(10., dBw/10.)


def watts2dBm(watts):
    return 10*np.log10(watts/1e-3)


def dB2lin(dB):
    return np.power(10.0, dB/10.0)


def lin2dB(lin):
    return 10.*np.log10(lin)


def pathloss(d2d):
    A_dB = -30
    d0 = 1
    d = d2d
    plos = dB2lin(A_dB) * np.power(d/d0, -3.5)
    
    return plos 

def calc_path_loss_los(fc_ghz, d_2d_meter):
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



def dbmtolin(dBm):
    return 10**(dBm/10)


def cb(rss):
    return 1/2*np.log(1+(rss/dB2lin(-30)))

def ce(rss):
    return 1/2*np.log(1+(rss/dB2lin(-30)))


def listOfTuples(list1, list2):
    return list(map(lambda x, y:(x,y), list1, list2))


def get_angle(vec0, vec1, unit="rad"):
    norm = np.linalg.norm
    angle = np.arccos(np.dot(vec0, vec1)/(norm(vec0) * norm(vec1)))
    assert unit in ["rad", "deg"], "The unit should be either rad (radian) or deg (degree)."
    angle = angle if unit == "rad" else np.rad2deg(angle)
    return angle