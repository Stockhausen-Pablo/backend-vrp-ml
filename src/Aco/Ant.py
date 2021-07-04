import pandas as pd

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
                 discountAlpha,
                 discountBeta,
                 pheromone_evaporation_coefficient,
                 firstInit=False):
        self.start_stop = start_stop
        self.microhub_hash = self.start_stop.hashIdentifier
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
        self.microhub_counter = 1
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
                            #self.possibleStops.remove(stop)
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
        self.microhub_counter += 1
        self.traverseAnt(self.current_stop, self.start_stop)
        self.allTours.append(self.tour)
        self.tour_complete = True

    def selectStop(self):
        print('-ant is selecting stop-')
        if self.firstInit:
            import random
            return random.choice(self.possibleStops)
            # while rnd == self.current_stop and len(self.possibleStops) > 1:
            #    rnd = random.choice(self.possibleStops)
            #    # self.firstInit = !firstInit
            # return rnd

        stopAttraction = dict()
        total_attraction = 0.0

        for possible_next_stop in self.possibleStops:
            current_hash = self.current_stop.hashIdentifier
            next_hash = possible_next_stop.hashIdentifier
            if current_hash == self.microhub_hash:
                current_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter)
            if next_hash == self.microhub_hash:
                next_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter + 1)
            print(current_hash)
            print(next_hash)
            print("-ant is retrieving pheromone-")
            if '{}/{}'.format(self.microhub_hash, self.microhub_counter) not in self.df_pheromoneMatrix.index.values:
                print("-ant filled missing microhub in df_pheromoneMatrix-")
                new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
                self.df_pheromoneMatrix = self.df_pheromoneMatrix.append(new_row, ignore_index=False)
                self.df_pheromoneMatrix['{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 0.0
                self.df_pheromoneMatrix.fillna(value=0.0, inplace=True)
            df_pheromoneValue = self.df_pheromoneMatrix.at[current_hash, next_hash]
            print("-ant is looking up distance-")
            distance = float(tManager.getDistanceByMatrix(self.current_stop.hashIdentifier, possible_next_stop.hashIdentifier))
            stopAttraction[possible_next_stop] = pow(df_pheromoneValue, self.discountAlpha) * pow(((1 / distance) if distance else 0),
                                                                                                  self.discountBeta)
            total_attraction += stopAttraction[possible_next_stop]

        if total_attraction == 0.0:
            for key, value in stopAttraction.items():
                stopAttraction[key] = self.next_up(value)
            total_attraction = self.next_up(total_attraction)

        return self.evaluateWeightChoices(stopAttraction, total_attraction)

    def next_up(self, x):
        import math
        import struct
        if math.isnan(x) or (math.isinf(x) and x > 0):
            return x
        if x == 0.0:
            x = 0.0
        n = struct.unpack('<q', struct.pack('<d', x))[0]
        if n >= 0:
            n += 1
        else:
            n -= 1
        return struct.unpack('<d', struct.pack('<q', n))[0]

    def evaluateWeightChoices(self, choices, total):
        print('-ant is evaluating weight choices-')
        import random
        from bisect import bisect
        r = random.uniform(0, total)
        upto = 0
        for key, value in choices.items():
            if upto + value >= r:
                return key
            upto += value
        assert False

    def traverseAnt(self, startStop, endStop):
        print('-ant is traversing between stops-')
        self.updateTour(endStop)
        self.updateDistanceTravelled(startStop, endStop)
        self.current_stop = endStop

    def startNewTour(self, microHub, temp_stops):
        print('-ant is starting a new tour-')
        self.microhub_counter += 1
        self.tour.append(microHub)
        self.allTours.append(self.tour.copy())
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
        print('-ant is resetting Tour-')
        self.tour = []
        self.tourOverloaded = 0
        self.tourWeight = 0.0
        self.tourVolume = 0.0

    def updateTour(self, newStopToAdd):
        print('-ant is updating Tour-')
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
