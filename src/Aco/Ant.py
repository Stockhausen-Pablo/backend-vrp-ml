from random import choices

import src.Tour.TourManager as tManager
from src.Utils.helper import linalg_norm_T


class Ant:
    def __init__(self, start_stop, current_stop, possibleStops, pheromoneMatrix, discountAlpha, discountBeta, firstInit=False):
        self.start_stop = start_stop
        self.possibleStops = possibleStops
        self.current_stop = current_stop
        self.tour = []
        self.tour_complete = False
        self.distance_travelled = 0.0
        self.discountAlpha = discountAlpha
        self.discountBeta = discountBeta
        self.pheromoneMatrix = pheromoneMatrix
        self.tour.append(self.start_stop)
        self.firstInit = firstInit
        self.possibleStops.remove(self.current_stop)

    def moveAnt(self):
        # Until Possible Locations is not Empty
        while self.possibleStops:
            next_stop = self.selectStop()
            self.traverseAnt(self.current_stop, next_stop)
        self.possibleStops.append(self.start_stop)
        self.traverseAnt(self.current_stop, self.start_stop)
        self.tour_complete = True

    def selectStop(self):
        if self.firstInit:
            import random
            rnd = random.choice(self.possibleStops)
            while rnd == self.current_stop and len(self.possibleStops) > 1:
                rnd = random.choice(self.possibleStops)
            return rnd

        stopAttraction = dict()
        total_attraction = 0.0

        for possible_next_stop in self.possibleStops:
            pheremoneValue = self.pheromoneMatrix[self.current_stop][possible_next_stop]
            distance = tManager.getDistance(self.current_stop, possible_next_stop)
            stopAttraction[possible_next_stop] = pow(pheremoneValue, self.discountAlpha) * pow((1 / distance),
                                                                                               self.discountBeta)
            total_attraction += stopAttraction[possible_next_stop]

        if total_attraction == 0:
            total_attraction = 1

        return choices(
            population=self.possibleStops,
            weights=[attraction / total_attraction for attraction in stopAttraction]
        )[0]

    def traverseAnt(self, startStop, endStop):
        self.updateTour(endStop)
        self.updateDistanceTravelled(startStop, endStop)
        self.current_stop = endStop

    def updateTour(self, newStopToAdd):
        self.tour.append(newStopToAdd)
        self.possibleStops = list(self.possibleStops)
        self.possibleStops.remove(newStopToAdd)

    def updateDistanceTravelled(self, startStop, endStop):
        self.distance_travelled += float(linalg_norm_T(startStop, endStop))

    def getTour(self):
        if self.tour_complete:
            return self.tour
        return None

    def get_travelled_Distance(self):
        if self.tour_complete:
            return self.distance_travelled
        return None
