import pandas as pd

import src.Tour.TourManager as tManager
from src.Utils.helper import calculate_distance


class Ant:
    def __init__(self,
                 start_stop,
                 current_stop,
                 ant_weight,
                 ant_volume,
                 possible_stops,
                 df_pheromone_matrix,
                 discount_alpha,
                 discount_beta,
                 pheromone_evaporation_coefficient,
                 first_run):

        # --------------------
        # STATES
        self.start_stop = start_stop
        self.microhub_hash = self.start_stop.hash_id
        self.possible_stops = possible_stops
        self.memory_possible_stops = []
        self.current_stop = current_stop

        # --------------------
        # ANT-SETTINGS
        self.ant_weight = ant_weight
        self.ant_volume = ant_volume

        # --------------------
        # TOUR-META
        self.all_tours = []
        self.tour = []
        self.tour_overload = 0
        self.tour_weight = 0.0
        self.tour_volume = 0.0
        self.tour_complete = False
        self.microhub_counter = 1

        # --------------------
        # ANT-META
        self.distance_travelled = 0.0  # kilometers
        self.discount_alpha = discount_alpha
        self.discount_beta = discount_beta
        self.df_pheromone_matrix = df_pheromone_matrix
        # self.pheromoneMatrix = pheromoneMatrix
        self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        # self.tour.append(self.start_stop)

        # --------------------
        # OTHER
        self.first_run = first_run
        self.updateTour(start_stop)

    def move_ant(self) -> object:
        """
        Moving ant until no possible stops are left.
        :rtype: object
        """
        print("-Ants started to move-")
        while self.possible_stops:
            next_stop = self.select_next_stop()
            possible_final_tour_weight = next_stop.demand_weight + self.tour_weight
            possible_final_tour_volume = next_stop.demand_volume + self.tour_volume
            temp_stops = self.possible_stops
            if (possible_final_tour_volume <= self.ant_volume) and (possible_final_tour_weight <= self.ant_weight):
                self.tour_overload = 0
                self.traverse_ant(self.current_stop, next_stop)
            else:
                if self.tour_overload >= (len(self.possible_stops) - 1):
                    stops_with_overload = []
                    for stop in self.possible_stops:
                        possible_final_tour_weight = stop.demand_weight + self.tour_weight
                        possible_final_tour_volume = stop.demand_volume + self.tour_volume
                        if (possible_final_tour_volume <= self.ant_volume) and (
                                possible_final_tour_weight <= self.ant_weight):
                            self.traverse_ant(self.current_stop, next_stop)
                            self.possible_stops = self.memory_possible_stops
                            # self.possibleStops.remove(stop)
                            continue
                        else:
                            stops_with_overload.append(stop)
                    if stops_with_overload:
                        self.start_new_tour(self.start_stop, temp_stops)
                        continue
                if self.tour_overload == 0:
                    self.memory_possible_stops = self.possible_stops.copy()
                    self.tour_overload += 1
                else:
                    self.possible_stops.remove(next_stop)
                    self.tour_overload += 1
        self.possible_stops.append(self.start_stop)
        self.microhub_counter += 1
        self.traverse_ant(self.current_stop, self.start_stop)
        self.all_tours.append(self.tour)
        self.tour_complete = True

    def select_next_stop(self) -> object:
        """
        Selecting next stop
        :rtype: object
        """
        print('-ant is selecting stop-')
        if self.first_run:
            import random
            return random.choice(self.possible_stops)
            # while rnd == self.current_stop and len(self.possibleStops) > 1:
            #    rnd = random.choice(self.possibleStops)
            #    # self.firstInit = !firstInit
            # return rnd

        stop_attraction = dict()
        total_attraction = 0.0

        for possible_next_stop in self.possible_stops:
            current_hash = self.current_stop.hash_id
            next_hash = possible_next_stop.hash_id
            if current_hash == self.microhub_hash:
                current_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter)
            if next_hash == self.microhub_hash:
                next_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter + 1)
            print("-ant is retrieving pheromone-")
            if '{}/{}'.format(self.microhub_hash, self.microhub_counter) not in self.df_pheromone_matrix.index.values:
                print("-ant filled missing microhub in df_pheromoneMatrix-")
                new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
                self.df_pheromone_matrix = self.df_pheromone_matrix.append(new_row, ignore_index=False)
                self.df_pheromone_matrix['{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 0.0
                self.df_pheromone_matrix.fillna(value=0.0, inplace=True)
            df_pheromone_value = self.df_pheromone_matrix.at[current_hash, next_hash]
            print("-ant is looking up distance-")
            distance = float(tManager.get_distance_by_matrix(self.current_stop.hash_id, possible_next_stop.hash_id))
            stop_attraction[possible_next_stop] = pow(df_pheromone_value, self.discount_alpha) * pow(
                ((1 / distance) if distance else 0),
                self.discount_beta)
            total_attraction += stop_attraction[possible_next_stop]

        if total_attraction == 0.0:
            for key, value in stop_attraction.items():
                stop_attraction[key] = self.define_threshold_for_next_stop(value)
            total_attraction = self.define_threshold_for_next_stop(total_attraction)

        return self.evaluate_weight_choices(stop_attraction, total_attraction)

    @staticmethod
    def define_threshold_for_next_stop(x: object) -> object:
        """
        Helper for determining the next stop threshold.
        """
        import math
        import struct
        if math.isnan(x) or (math.isinf(x) and x > 0):
            return x
        if x == 0.0:
            x = 0.0
        n = struct.unpack('<q', struct.pack('<d', x))[0]  # transform to int
        if n >= 0:
            n += 1
        else:
            n -= 1
        return struct.unpack('<d', struct.pack('<q', n))[0]

    @staticmethod
    def evaluate_weight_choices(choices: object, total: object) -> object:
        """
        Evaluate weights of each possible next stop and chooses based on the defined threshold.
        :param choices: dict of all next stops
        :param total: threshold for choosing next stop
        :return: next stop hash_id
        """
        print('-ant is evaluating weight choices-')
        import random
        r = random.uniform(0, total)
        current_value_sum = 0
        for key, value in choices.items():
            if current_value_sum + value >= r:
                return key
            current_value_sum += value
        assert False

    def traverse_ant(self, startStop: object, endStop: object) -> object:
        """
        Let Ant traverse between two points.
        :param startStop: point of departure
        :param endStop: final point
        """
        print('-ant is traversing between stops-')
        self.updateTour(endStop)
        self.update_distance_travelled(startStop, endStop)
        self.current_stop = endStop

    def start_new_tour(self, microHub: object, temp_stops: object) -> object:
        """
        Starts a new Tour and updates tour meta.
        :param microHub: Microhub as the stop object.
        :param temp_stops: List of temporary stops to not lose track of last possible stop.
        :return:
        """
        print('-ant is starting a new tour-')
        self.microhub_counter += 1
        self.tour.append(microHub)
        self.all_tours.append(self.tour.copy())
        self.update_distance_travelled(self.current_stop, microHub)
        self.current_stop = microHub
        self.reset_tour()
        self.tour.append(microHub)
        if not self.memory_possible_stops:
            self.possible_stops = temp_stops
        else:
            self.possible_stops = self.memory_possible_stops
        # self.possibleStops = list(self.possibleStops)
        # self.possibleStops.remove(microHub)

    def reset_tour(self) -> object:
        """
        Resets tour and tour meta data.
        """
        print('-ant is resetting Tour-')
        self.tour = []
        self.tour_overload = 0
        self.tour_weight = 0.0
        self.tour_volume = 0.0

    def updateTour(self, newStopToAdd):
        print('-ant is updating Tour-')
        self.tour.append(newStopToAdd)
        self.tour_weight += newStopToAdd.demand_weight
        self.tour_volume += newStopToAdd.demand_volume
        self.possible_stops = list(self.possible_stops)
        self.possible_stops.remove(newStopToAdd)
        self.memory_possible_stops = []

    def update_distance_travelled(self, startStop: object, endStop: object) -> object:
        """
        Update the tour meta data entity travelled distance.
        :param startStop: point of departure
        :param endStop: final point
        :return: updated distance travelled
        """
        self.distance_travelled += float(calculate_distance(startStop, endStop))

    def get_all_tours(self) -> object:
        """
        :return: List of all tours
        """
        if self.tour_complete:
            return self.all_tours
        return None

    def get_travelled_distance(self) -> object:
        """
        :return: Overall travelled distance
        """
        if self.tour_complete:
            return self.distance_travelled
        return None
