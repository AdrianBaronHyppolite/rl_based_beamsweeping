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


def pathloss(loc0, loc1, alpha):
    A_dB = -30
    d0 = 1
    d = np.linalg.norm(loc0-loc1)
    return dB2lin(A_dB) * np.power(d/d0, -alpha)


def listOfTuples(list1, list2):
    return list(map(lambda x, y:(x,y), list1, list2))


def get_angle(vec0, vec1, unit="rad"):
    norm = np.linalg.norm
    angle = np.arccos(np.dot(vec0, vec1)/(norm(vec0) * norm(vec1)))
    assert unit in ["rad", "deg"], "The unit should be either rad (radian) or deg (degree)."
    angle = angle if unit == "rad" else np.rad2deg(angle)
    return angle