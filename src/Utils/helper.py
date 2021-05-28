import numpy as np
import math


def linalg_norm_T(startStop, endStop):
    a = [startStop.longitude, startStop.latitude]
    b = [endStop.longitude, endStop.latitude]
    distance = np.linalg.norm(np.array(a) - np.array(b), axis=0) * 100
    return distance * math.sqrt(2)


def rectified(x):
    return max(0.0, x)


def normalize_list(probList):
    s = sum(probList)
    normRow = [float(i) / s for i in probList]
    return normRow


def normalize_df(df):
    for row in df:
        toNormalizeRow = df[row].to_numpy()
        s = sum(toNormalizeRow)
        normRow = [float(i) / s for i in toNormalizeRow]
        df.loc[row] = normRow
    return df


def activationBySoftmax(X):
    z = np.exp(np.array((X - max(X)), float))
    y = np.sum(z)
    return z / y
