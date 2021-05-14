from typing import List
from src.Tour.Stop import Stop

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
            longitude_diff = abs(start.getStop()[0] - end.getStop()[0])
            latitude_diff = abs(start.getStop()[1] - end.getStop()[1])
            distances_from_start.append((longitude_diff ** 2 + latitude_diff ** 2) ** 0.5)

        distances.append(distances_from_start)


def getListOfStops():
    return stops


def getDistance(index_a: int, index_b: int) -> float:
    return distances[index_a][index_b]


def clear():
    stops.clear()
    distances.clear()
