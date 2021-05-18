import math
import struct

import src.Tour.TourManager as tManager
from src.Utils.helper import linalg_norm_T


class Ant:
    def __init__(self,
                 start_stop,
                 current_stop,
                 antWeight,
                 antVolume,
                 possibleStops,
                 df_pheromoneMatrix,
                 # pheromoneMatrix,
                 discountAlpha,
                 discountBeta,
                 pheromone_evaporation_coefficient,
                 firstInit=False):
        self.start_stop = start_stop
        self.possibleStops = possibleStops
        self.current_stop = current_stop
        self.antWeight = antWeight
        self.antVolume = antVolume
        self.rmdPossibleStops = []
        self.allTours = []
        self.tour = []
        self.tourOverloaded = 0
        self.tourWeight = 0.0
        self.tourVolume = 0.0
        self.tour_complete = False
        self.distance_travelled = 0.0  # kilometers
        self.discountAlpha = discountAlpha
        self.discountBeta = discountBeta
        self.df_pheromoneMatrix = df_pheromoneMatrix
        # self.pheromoneMatrix = pheromoneMatrix
        self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        # self.tour.append(self.start_stop)
        self.firstInit = firstInit
        self.updateTour(start_stop)
        # self.possibleStops.remove(self.current_stop)

    def moveAnt(self):
        print("-Ants started to move-")
        # Until Possible Locations is not Empty
        while self.possibleStops:
            next_stop = self.selectStop()
            possible_final_tourWeight = next_stop.demandWeight + self.tourWeight
            possible_final_tourVolume = next_stop.demandVolume + self.tourVolume
            temp_stops = self.possibleStops
            if ((possible_final_tourVolume <= self.antVolume) and (possible_final_tourWeight <= self.antWeight)):
                self.tourOverloaded = 0
                self.traverseAnt(self.current_stop, next_stop)
            else:
                if (self.tourOverloaded >= (len(self.possibleStops) - 1)):
                    overloadedStops = []
                    for stop in self.possibleStops:
                        possible_final_tourWeight = stop.demandWeight + self.tourWeight
                        possible_final_tourVolume = stop.demandVolume + self.tourVolume
                        if ((possible_final_tourVolume <= self.antVolume) and (
                            possible_final_tourWeight <= self.antWeight)):
                            self.traverseAnt(self.current_stop, next_stop)
                            self.possibleStops = self.rmdPossibleStops
                            self.possibleStops.remove(stop)
                            continue
                        else:
                            overloadedStops.append(stop)
                    if overloadedStops:
                        self.startNewTour(self.start_stop, temp_stops)
                        continue
                if (self.tourOverloaded == 0):
                    self.rmdPossibleStops = self.possibleStops.copy()
                    self.tourOverloaded += 1
                else:
                    self.possibleStops.remove(next_stop)
                    self.tourOverloaded += 1
        self.possibleStops.append(self.start_stop)
        self.traverseAnt(self.current_stop, self.start_stop)
        self.allTours.append(self.tour)
        self.tour_complete = True

    def selectStop(self):
        if self.firstInit:
            import random
            rnd = random.choice(self.possibleStops)
            while rnd == self.current_stop and len(self.possibleStops) > 1:
                rnd = random.choice(self.possibleStops)
                # self.firstInit = !firstInit
            return rnd

        stopAttraction = dict()
        total_attraction = 0.0

        for possible_next_stop in self.possibleStops:
            df_pheromoneValue = self.df_pheromoneMatrix.at[
                self.current_stop.hashIdentifier, possible_next_stop.hashIdentifier]
            # pheromoneValue = self.pheromoneMatrix[self.current_stop.stopid][possible_next_stop.stopid]
            # self.pheromoneMatrix[self.current_stop.stopid][self.current_stop.stopid] = 0.0
            distance = tManager.getDistance(self.current_stop.stopid, possible_next_stop.stopid)
            stopAttraction[possible_next_stop] = pow(df_pheromoneValue, self.discountAlpha) * pow((1 / distance),
                                                                                                  self.discountBeta)
            total_attraction += stopAttraction[possible_next_stop]

        if total_attraction == 0:
            for key, value in stopAttraction.items():
                if math.isnan(value) or (math.isinf(value) and value > 0):
                    stopAttraction[key] = value
                if value == 0.0:
                    stopAttraction[key] = 0.0
                n = struct.unpack('<q', struct.pack('<d', value))[0]
                if n >= 0:
                    n += 1
                else:
                    n -= 1
                stopAttraction[key] = struct.unpack('<d', struct.pack('<q', n))[0]
        import random
        rndFactor = random.random()

        cummulative = 0
        for possible_next_location in stopAttraction:
            weight = (stopAttraction[possible_next_location] / total_attraction)
            if rndFactor <= weight + cummulative:
                return possible_next_location
            cummulative += weight

    def traverseAnt(self, startStop, endStop):
        self.updateTour(endStop)
        self.updateDistanceTravelled(startStop, endStop)
        self.current_stop = endStop

    def handleLastStopTour(self, lastStop):
        self.tour.append(self.start_stop)
        self.allTours.append(self.tour)
        self.updateDistanceTravelled(self.current_stop, self.start_stop)
        self.resetTour()
        self.tour.append(self.start_stop)
        self.current_stop = self.start_stop
        self.tour.append(lastStop)
        self.updateDistanceTravelled(self.current_stop, lastStop)
        self.current_stop = lastStop

    def startNewTour(self, microHub, temp_stops):
        self.tour.append(microHub)
        self.allTours.append(self.tour)
        self.updateDistanceTravelled(self.current_stop, microHub)
        self.current_stop = microHub
        self.resetTour()
        self.tour.append(microHub)
        if not self.rmdPossibleStops:
            self.possibleStops = temp_stops
        else:
            self.possibleStops = self.rmdPossibleStops
        # self.possibleStops = list(self.possibleStops)
        # self.possibleStops.remove(microHub)

    def resetTour(self):
        self.tour = []
        self.tourOverloaded = 0
        self.tourWeight = 0.0
        self.tourVolume = 0.0

    def updateTour(self, newStopToAdd):
        self.tour.append(newStopToAdd)
        self.tourWeight += newStopToAdd.demandWeight
        self.tourVolume += newStopToAdd.demandVolume
        self.possibleStops = list(self.possibleStops)
        self.possibleStops.remove(newStopToAdd)
        self.rmdPossibleStops = []

    def updateDistanceTravelled(self, startStop, endStop):
        self.distance_travelled += float(linalg_norm_T(startStop, endStop))

    def getTour(self):
        if self.tour_complete:
            return self.tour
        return None

    def getTours(self):
        if self.tour_complete:
            return self.allTours
        return None

    def get_travelled_Distance(self):
        if self.tour_complete:
            return self.distance_travelled
        return None
