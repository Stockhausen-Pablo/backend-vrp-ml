from random import randint
from typing import Tuple


class Stop:
    def __init__(self,
                 hash_id: str = None,
                 stop_id: int = None,
                 longitude: float = None,
                 latitude: float = None,
                 demand_weight: float = None,
                 demand_volume: float = None,
                 box_amount: int = None,
                 tourStopId: int = None) -> None:
        self.hash_id = hash_id
        self.stop_id = stop_id
        self.longitude = longitude if longitude else randint(0, 1000)
        self.latitude = latitude if latitude else randint(0, 1000)
        self.demand_weight = demand_weight
        self.demand_volume = demand_volume
        self.box_amount = box_amount
        self.tourStopId = tourStopId  # za property -> keep name

    def getStop(self) -> Tuple[str, int, float, float, float, float, int, int]:
        """
        :return: stop properties
        """
        return self.hash_id, self.stop_id, self.longitude, self.latitude, self.demand_weight, self.demand_volume, self.box_amount, self.tourStopId
