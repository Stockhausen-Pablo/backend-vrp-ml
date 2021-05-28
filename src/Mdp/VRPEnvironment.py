class VRPEnvironment:
    """
    MDP-Model" is provided inside code for the environment
        a predictive model of the env that resolves probabilities of next reward and next state following an action from a state
    """

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

    def step(self, action, suggested_next_state):
        if action == 0:
            return self.evaluate_action_0()

        if action == 1:
            return self.evaluate_action_1(suggested_next_state)

        if action == 2:
            return self.evaluate_action_2()

    def evaluate_action_0(self):
        next_state = self.getMicrohub()
        reward = self.reward_func(self.current_state, next_state)
        self.current_tour.append(next_state)
        self.current_state = next_state
        self.resetTour()
        return next_state, reward, self.done, self.current_tour, self.allTours

    def evaluate_action_1(self, suggested_next_state):
        next_state = self.getStateByHash(suggested_next_state)
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

    def possible_rewards(self, state, action_space_list):
        possible_rewards = self.distanceMatrix.loc[state, action_space_list]
        return possible_rewards

    def getLegalAction(self):
        legal_next_states = []
        for stop in self.possibleStops:
            possible_tour_weight = float(stop.demandWeight) + self.current_tour_weight
            possible_tour_volume = float(stop.demandVolume) + self.current_tour_volume
            if (possible_tour_weight <= self.vehicleWeight and possible_tour_volume <= self.vehicleVolume):
                legal_next_states.append(stop.hashIdentifier)

        if legal_next_states:
            action = 1
            return action, legal_next_states

        if not legal_next_states and not self.possibleStops:
            legal_next_states.append(self.microHub.hashIdentifier)
            action = 2
            return action, legal_next_states

        if not legal_next_states and self.possibleStops:
            legal_next_states.append(self.microHub.hashIdentifier)
            action = 0
            return action, legal_next_states
