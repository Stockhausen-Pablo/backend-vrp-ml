import numpy as np


def linalg_norm_T(startStop, endStop):
    a = [startStop.longitude, startStop.latitude]
    b = [endStop.longitude, endStop.latitude]
    return np.linalg.norm(np.array(a) - np.array(b), axis=0) * 100
