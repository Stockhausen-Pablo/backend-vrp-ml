from random import randint
from typing import Tuple


class Stop:
    def __init__(self, hashIdentifier: float = None, stopid: int = None, longitude: float = None, latitude: float = None, demandWeight: int = None, demandVolume: int = None) -> None:
        self.hashIdentifier = hashIdentifier
        self.stopid = stopid
        self.longitude = longitude if longitude else randint(0, 1000)
        self.latitude = latitude if latitude else randint(0, 1000)
        self.demandWeight = demandWeight if demandWeight else randint(5, 20)
        self.demandVolume = demandVolume if demandVolume else randint(20, 50)

    def getStop(self) -> Tuple[float, int, float, float, int, int]:
        return self.hashIdentifier, self.stopid, self.longitude, self.latitude, self.demandWeight, self.demandVolume
