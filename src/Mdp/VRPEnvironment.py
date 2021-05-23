class VRPEnvironment:

    def __init__(self, states, actions, probabilityMatrix, distanceMatrix, rewardFunction, microHub, capacityDemands,
                 vehicles, vehicleWeight, vehicleVolume):
        self.states = states
        self.actions = actions
        self.probabilityMatrix = probabilityMatrix
        self.distanceMatrix = distanceMatrix
        self.rewardFunction = rewardFunction
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
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

    def reward_func(self, current_stop, next_stop):
        reward = self.distanceMatrix.at[current_stop.hashIdentifier, next_stop.hashIdentifier]
        return reward

    def reward_func_hash(self, current_stop, next_stop):
        return self.distanceMatrix.at[current_stop, next_stop]

    def getTransitionProbability(self, stop_i_hash, stop_j_hash):
        return self.probabilityMatrix.at[stop_i_hash, stop_j_hash]

    def getCapacityDemandOfStop(self, stop_hash):
        return list(self.capacityDemands[stop_hash].values())

    def getMicrohubId(self):
        return self.microHub.hashIdentifier

    def getStateByHash(self, hashIdentifier):
        return next((state for state in self.states if state.hashIdentifier == hashIdentifier), None)

    def getStateHashes(self):
        return [state.hashIdentifier for state in self.states]

    def getLegalActions(self):
        legalActions = []
        for stop in self.possibleStops:
            possible_tour_weight = float(stop.demandWeight) + self.current_tour_weight
            possible_tour_volume = float(stop.demandVolume) + self.current_tour_volume
            if (possible_tour_weight <= self.vehicleWeight and possible_tour_volume <= self.vehicleVolume):
                legalActions.append(stop.hashIdentifier)

        if not legalActions:
            legalActions.append(self.microHub.hashIdentifier)
        return legalActions
