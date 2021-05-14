from random import randint
from typing import Tuple


class Stop:
    def __init__(self, longitude: float = None, latitude: float = None) -> None:
        self.longitude = longitude if longitude else randint(0, 1000)
        self.latitude = latitude if latitude else randint(0, 1000)

    def getStop(self) -> Tuple[float, float]:
        return self.longitude, self.latitude
