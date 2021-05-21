import numpy as np


def linalg_norm_T(startStop, endStop):
    a = [startStop.longitude, startStop.latitude]
    b = [endStop.longitude, endStop.latitude]
    return np.linalg.norm(np.array(a) - np.array(b), axis=0) * 100

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


def applySoftmax(df_probabilityMatrix):
    for row in df_probabilityMatrix:
        toNormalizeRow = df_probabilityMatrix[row].to_numpy()
        softmaxRow = softmax(toNormalizeRow)
        df_probabilityMatrix[row] = softmaxRow
    return df_probabilityMatrix


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
