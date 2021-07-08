import numpy as np
import math


def calculate_tour_meta(vehicle_speed, stay_duration, final_tours):
    """
    Calculates the tour meta data for the constructed tours.
    ----------
    Returns the total time and distance required for all tours aswell as the mean values.
    """
    total_time = 0.0
    total_distance = 0.0
    for tour in final_tours:
        for idx, stop in enumerate(tour):
            current_stop = stop
            next_stop = tour[(idx + 1) % len(tour)]
            distance = calculate_distance(current_stop, next_stop)
            time_estimated = ((distance / vehicle_speed) * 60) + stay_duration if distance > 0.0 else 0.0
            total_time += time_estimated
            total_distance += distance

    average_time_per_tour = total_time / len(final_tours)
    average_distance_per_tour = total_distance / len(final_tours)

    return total_time, total_distance, average_time_per_tour, average_distance_per_tour


def calculate_distance(startStop, endStop):
    """
    Calculates the distance between two coordinates following the Haversine Formula.
    """
    R = 6373.0
    lat1 = math.radians(startStop.latitude)
    lon1 = math.radians(startStop.longitude)
    lat2 = math.radians(endStop.latitude)
    lon2 = math.radians(endStop.longitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.sin(dlon / 2) * math.sin(dlon / 2) * math.cos(lat1) * math.cos(
        lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    distance_approximation = distance * np.sqrt(2)
    return distance_approximation


def normalize_list(probList):
    """
    Normalizes a given list, that way the sum of all values sum up to 1.
    """
    if min(probList) == 0.0 and max(probList) == 0.0:
        probList += 1 / len(probList)
        return probList
    s = sum(probList)
    normRow = [float(i) / s for i in probList]
    return normRow

# Not being used.
def normalize_df(df):
    for row in df:
        toNormalizeRow = df[row].to_numpy()
        s = sum(toNormalizeRow)
        normRow = [float(i) / s for i in toNormalizeRow]
        df.loc[row] = normRow
    return df


def activation_by_softmax(X):
    """
    Compute the softmax of each element along an axis of X.
    Parameters
    ----------
    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    z = np.exp(np.array((X - max(X)), float))
    y = np.sum(z)

    return z / y


def action_by_softmax_as_dict(X):
    """
    Compute the softmax of each element along an axis of X.
    Parameters
    ----------
    Returns an dict the same size as X with (hashIdentifier-value)-pairs. The result will sum to 1
    along the specified axis.
    """
    to_softmax_series = X.copy()
    for state_hash, prob in to_softmax_series.items():
        to_softmax_series[state_hash] = prob
    z = np.exp(np.array((to_softmax_series - max(to_softmax_series)), float))
    y = np.sum(z)
    softmax_arr = z / y
    softmax_dict = dict()
    counter = 0
    for state_hash, prob in X.items():
        softmax_dict[state_hash] = softmax_arr[counter]
        counter += 1
    return softmax_dict
