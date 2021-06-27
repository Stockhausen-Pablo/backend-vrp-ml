from random import randint
from typing import Tuple


class Stop:
    def __init__(self, hashIdentifier: str = None, stopid: int = None, longitude: float = None, latitude: float = None, demandWeight: float = None, demandVolume: float = None, boxAmount: int = None, tourStopId: int = None) -> None:
        self.hashIdentifier = hashIdentifier
        self.stopid = stopid
        self.longitude = longitude if longitude else randint(0, 1000)
        self.latitude = latitude if latitude else randint(0, 1000)
        self.demandWeight = demandWeight
        self.demandVolume = demandVolume
        self.boxAmount = boxAmount
        self.tourStopId = tourStopId

    def getStop(self) -> Tuple[str, int, float, float, float, float, int, int]:
        return self.hashIdentifier, self.stopid, self.longitude, self.latitude, self.demandWeight, self.demandVolume, self.boxAmount, self.tourStopId
