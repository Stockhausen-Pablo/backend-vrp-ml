import numpy as np
from src.Utils.helper import normalize_list


class VRPEnvironment:

    def __init__(self, states, actions, probabilityMatrix, distanceMatrix, microHub, capacityDemands,
                 vehicles, vehicleWeight, vehicleVolume):
        self.states = states
        self.actions = actions
        self.probabilityMatrix = probabilityMatrix
        self.distanceMatrix = distanceMatrix
        self.microHub = microHub

        # demands
        self.vehicles = vehicles
        self.capacityDemands = capacityDemands
        self.vehicleWeight = vehicleWeight
        self.vehicleVolume = vehicleVolume

        # current
        self.current_state = None
        self.current_state = [self.microHub if self.current_state is None else self.current_state]
        self.possibleStops = []
        self.done = False

        # current Tours
        self.allTours = []
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

        # init
        self.resetPossibleStops()

    def reset(self):
        self.done = False
        self.resetPossibleStops()
        self.resetTours()
        return self.current_state

    def takeStep(self, action):
        legal_next_states, legal_action = self.getLegalActions()

        if legal_action != action:
            reward = 5000
            return self.current_state, reward, self.done, self.current_tour, self.allTours
        else:
            if action == 0:
                return self.evaluate_action_0()

            if action == 1:
                return self.evaluate_action_1(legal_next_states)

            if action == 2:
                return self.evaluate_action_2()

    def evaluate_action_0(self):
        next_state = self.getMicrohub()
        reward = self.reward_func(self.current_state, next_state)
        self.current_tour.append(next_state)
        self.current_state = next_state
        self.resetTour()
        return next_state, reward, self.done, self.current_tour, self.allTours

    def evaluate_action_1(self, legal_next_states):
        next_states_prob = self.probabilityMatrix.loc[self.current_state.hashIdentifier, legal_next_states].to_numpy()
        next_states_prob = normalize_list(next_states_prob) if len(legal_next_states) > 1 else [1.0]
        next_state_hash = np.random.choice(legal_next_states, p=next_states_prob)
        next_state = self.getStateByHash(next_state_hash)
        reward = self.reward_func(self.current_state, next_state)
        self.current_state = next_state
        self.updateTourMeta(next_state)
        return next_state, reward, self.done, self.current_tour, self.allTours

    def evaluate_action_2(self):
        next_state = self.getMicrohub()
        reward = self.reward_func(self.current_state, next_state)
        self.current_tour.append(next_state)
        self.current_state = next_state
        self.done = True
        self.resetTour()
        return next_state, reward, self.done, self.current_tour, self.allTours

    def step(self, action, actionNr):
        # Check which action got selected : 0 = back to Depot ; 1 = select unvisited Node
        if action == 0:
            print("--Action 0 was selected")
            print("--The last stop of the tour will be the Microhub and the tour is completed.")
            if not self.possibleStops:
                self.done = True
            reward = self.reward_func(self.current_state, self.microHub)
            self.current_tour.append(self.microHub)
            self.allTours.append(self.current_tour)
            self.current_state = self.microHub
            self.resetTour()
            return self.microHub, reward, self.done, self.current_tour, self.allTours
        else:
            print("--Action 1 was selected")
            print("--The next available node will be selected.")
            relevantStop = next((x for x in self.possibleStops if x.hashIdentifier == actionNr), None)
            reward = self.reward_func(self.current_state, relevantStop)
            if not self.possibleStops:
                self.allTours.append(self.current_tour)
                self.done = True
            self.possibleStops.remove(relevantStop)
            self.current_tour.append(relevantStop)
            self.current_tour_weight += relevantStop.demandWeight
            self.current_tour_volume += relevantStop.demandVolume
            return relevantStop, reward, self.done, self.current_tour, self.allTours

    def predict(self, state_i, legalActions):
        listPredictions = []
        for state_j in self.probabilityMatrix:
            if state_j in legalActions:
                prediction = self.probabilityMatrix.at[state_i.hashIdentifier, state_j]
                listPredictions.append(prediction)
        return listPredictions

    def resetPossibleStops(self):
        copyStates = self.states.copy()
        copyStates.remove(self.microHub)
        self.current_state = self.microHub
        self.possibleStops = copyStates

    def resetTours(self):
        self.allTours = []
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

    def resetTour(self):
        self.allTours.append(self.current_tour)
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

    def updateTourMeta(self, next_state):
        self.possibleStops.remove(next_state)
        self.current_tour.append(next_state)
        self.current_tour_weight += next_state.demandWeight
        self.current_tour_volume += next_state.demandVolume

    def reward_func(self, current_stop, next_stop):
        reward = self.distanceMatrix.at[current_stop.hashIdentifier, next_stop.hashIdentifier]
        return reward

    def reward_func_hash(self, current_stop, next_stop):
        return self.distanceMatrix.at[current_stop, next_stop]

    def getTransitionProbability(self, stop_i_hash, stop_j_hash):
        return self.probabilityMatrix.at[stop_i_hash, stop_j_hash]

    def getCapacityDemandOfStop(self, stop_hash):
        return list(self.capacityDemands[stop_hash].values())

    def getMicrohub(self):
        return self.microHub

    def getStateByHash(self, hashIdentifier):
        return next((state for state in self.states if state.hashIdentifier == hashIdentifier), None)

    def getStateHashes(self):
        return [state.hashIdentifier for state in self.states]

    def getLegalActions(self):
        legal_next_states = []
        action = 0
        for stop in self.possibleStops:
            possible_tour_weight = float(stop.demandWeight) + self.current_tour_weight
            possible_tour_volume = float(stop.demandVolume) + self.current_tour_volume
            if (possible_tour_weight <= self.vehicleWeight and possible_tour_volume <= self.vehicleVolume):
                legal_next_states.append(stop.hashIdentifier)

        if legal_next_states:
            action = 1

        if not legal_next_states and not self.possibleStops:
            legal_next_states.append(self.microHub.hashIdentifier)
            action = 2

        if not legal_next_states and self.possibleStops:
            legal_next_states.append(self.microHub.hashIdentifier)
            action = 0

        return legal_next_states, action
