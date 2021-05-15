from random import randint
from typing import Tuple


class Stop:
    def __init__(self, stopid: int = None, longitude: float = None, latitude: float = None, demand: int = None) -> None:
        self.stopid = stopid
        self.longitude = longitude if longitude else randint(0, 1000)
        self.latitude = latitude if latitude else randint(0, 1000)
        self.demand = demand if demand else randint(5, 20)

    def getStop(self) -> Tuple[int, float, float, int]:
        return self.stopid, self.longitude, self.latitude, self.demand
