from typing import List

import pandas as pd

from src.Tour.Stop import Stop
from src.Utils.helper import linalg_norm_T

stops: List[Stop] = []
distances: List[List[float]] = []

def setup_distanceMatrix():
    df_new_distanceMatrix = pd.DataFrame(index=[stop.hashIdentifier for stop in stops],
                                         columns=[stop.hashIdentifier for stop in stops])
    df_new_distanceMatrix.fillna(value=0.0, inplace=True)
    return df_new_distanceMatrix


distanceMatrix= setup_distanceMatrix()
capacityDemands = dict()


def addStop(stop: Stop) -> None:
    stops.append(stop)


def getStop(index: int) -> Stop:
    return stops[index]


def getStopByHash(id: float) -> Stop:
    return [x for x in stops if x.hashIdentifier == id]


def getMicrohub():
    return stops[0]


def getLength() -> int:
    return len(stops)


def calculateDistances() -> None:
    for start in stops:
        distances_from_start = []
        for end in stops:
            distances_from_start.append(linalg_norm_T(start, end))
        distances.append(distances_from_start)


def calculateDistanceMatrix() -> None:
    for stop_i in stops:
        for stop_j in stops:
            distance_i_j = linalg_norm_T(stop_i, stop_j)
            distanceMatrix.at[stop_i.hashIdentifier, stop_j.hashIdentifier] = distance_i_j


def initCapacityDemands() -> None:
    for stop in stops:
        capacityDemands[stop.hashIdentifier] = [stop.demandWeight, stop.demandVolume]


def getListOfStops():
    return stops


def getDistances():
    return distanceMatrix


def getDistance(index_a: int, index_b: int) -> float:
    return distances[index_a][index_b]


def getCapacityDemands():
    return capacityDemands


def clear():
    stops.clear()
    distances.clear()
