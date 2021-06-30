import numpy as np
import math


def calculate_delivery_time(vehicle_speed, stay_duration, final_tours):
    total_time = 0.0
    total_distance = 0.0
    for tour in final_tours:
        for idx, stop in enumerate(tour):
            current_stop = stop
            next_stop = tour[(idx + 1) % len(tour)]
            distance = linalg_norm_T(current_stop, next_stop)
            time_estimated = ((distance / vehicle_speed) * 60) + stay_duration if distance > 0.0 else 0.0
            total_time += time_estimated
            total_distance += distance

    average_time_per_tour = total_time / len(final_tours)

    return total_time, total_distance, average_time_per_tour


def calculate_meta_for_za_tour(vehicle_speed, stay_duration, final_tours):
    total_time = 0.0
    total_distance = 0.0
    for tour in final_tours:
        for idx, stop in enumerate(tour):
            current_stop = stop
            next_stop = tour[(idx + 1) % len(tour)]
            distance = linalg_norm_T(current_stop, next_stop)
            time_estimated = ((distance / vehicle_speed) * 60) + stay_duration if distance > 0.0 else 0.0
            total_time += time_estimated
            total_distance += distance

    average_time_per_tour = total_time / len(final_tours)
    average_distance_per_tour = total_distance / len(final_tours)

    return total_time, total_distance, average_time_per_tour, average_distance_per_tour


def linalg_norm_T(startStop, endStop):
    # haversine formula
    R = 6373.0
    lat1 = math.radians(startStop.latitude)
    lon1 = math.radians(startStop.longitude)
    lat2 = math.radians(endStop.latitude)
    lon2 = math.radians(endStop.longitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) * math.sin(dlat/2) + math.sin(dlon/2) * math.sin(dlon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R*c
    distance_approximation = distance * np.sqrt(2)
    return distance_approximation


def rectified(x):
    return max(0.0, x)


def normalize_list(probList):
    if min(probList) == 0.0 and max(probList) == 0.0:
        probList += 1 / len(probList)
        return probList
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


def softmaxDict(X):
    toSoftmaxSeries = X.copy()
    for state_hash, prob in toSoftmaxSeries.items():
        toSoftmaxSeries[state_hash] = prob
    z = np.exp(np.array((toSoftmaxSeries - max(toSoftmaxSeries)), float))
    y = np.sum(z)
    softmax_arr = z / y
    softmax_dict = dict()
    counter = 0
    for state_hash, prob in X.items():
        softmax_dict[state_hash] = softmax_arr[counter]
        counter += 1
    return softmax_dict


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
