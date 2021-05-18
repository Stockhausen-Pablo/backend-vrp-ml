from typing import List

from src.Tour.Stop import Stop
from src.Utils.helper import linalg_norm_T

stops: List[Stop] = []
distances: List[List[float]] = []


def addStop(stop: Stop) -> None:
    stops.append(stop)


def getStop(index: int) -> Stop:
    return stops[index]


def getLength() -> int:
    return len(stops)


def calculateDistances() -> None:
    for start in stops:
        distances_from_start = []
        for end in stops:
            distances_from_start.append(linalg_norm_T(start, end))
        distances.append(distances_from_start)


def getListOfStops():
    return stops


def getListOfDistances():
    return distances


def getDistance(index_a: int, index_b: int) -> float:
    return distances[index_a][index_b]


def clear():
    stops.clear()
    distances.clear()
