class VRPEnvironment:
    def __init__(self,
                 states,
                 actions,
                 distance_matrix,
                 microhub,
                 capacity_demands,
                 vehicles,
                 vehicle_weight,
                 vehicle_volume):

        # --------------------
        # STATES / ACTIONS
        self.states = states
        self.actions = actions
        self.distance_matrix = distance_matrix
        self.microhub = microhub
        self.microhub_counter = 0

        # --------------------
        # DEMANDS
        self.vehicles = vehicles
        self.capacity_demands = capacity_demands
        self.vehicle_weight = vehicle_weight
        self.vehicle_volume = vehicle_volume

        # --------------------
        # META/MICRO INFORMATION
        self.current_state = None
        self.current_state = [self.microhub if self.current_state is None else self.current_state]
        self.possible_stops = []
        self.done = False

        # --------------------
        # TOURS META
        self.all_tours = []
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

        # --------------------
        # ON INIT
        self.reset_possible_stops()

    def reset(self) -> object:
        """
        Resets whole environment to initialization state.
        :return: starting state (depot)
        """
        self.done = False
        self.reset_possible_stops()
        self.reset_tours()
        self.microhub_counter = 0
        return self.current_state

    def step(self, action: object, suggested_next_state: object) -> object:
        """
        Does a step in the environment following an given action by the agent.
        :param action: choosed action by the agent
        :param suggested_next_state: suggested next state by the agent
        :return: next_state, reward, boolean done, current_tour, all_tours
        """
        if action == 0:
            return self.evaluate_action_0()

        if action == 1:
            return self.evaluate_action_1(suggested_next_state)

        if action == 2:
            return self.evaluate_action_2()

    def evaluate_action_0(self) -> object:
        """
        Evaluates action 0, that the next stop will be the microhub but there are still possible stops left.
        :return: next_state, reward, boolean done, current_tour, all_tours
        """
        next_state = self.get_microhub()
        reward = self.reward_func(self.current_state, next_state)
        self.current_tour.append(next_state)
        self.current_state = next_state
        self.reset_tour()
        return next_state, reward, self.done, self.current_tour, self.all_tours

    def evaluate_action_1(self, suggested_next_state: object) -> object:
        """
        Evaluates action 1, that the next stop will be one of the possible stops but there are still more possible stops left.
        :param suggested_next_state: suggested next state by the agent
        :return: next_state, reward, boolean done, current_tour, all_tours
        """
        next_state = self.get_state_by_hash(suggested_next_state)
        reward = self.reward_func(self.current_state, next_state)
        self.current_state = next_state
        self.update_tour_meta(next_state)
        return next_state, reward, self.done, self.current_tour, self.all_tours

    def evaluate_action_2(self) -> object:
        """
        Evaluates action 2, that the next stop will be the microhub and there no possible stops left.
        :return: next_state, reward, boolean done, current_tour, all_tours
        """
        next_state = self.get_microhub()
        reward = self.reward_func(self.current_state, next_state)
        self.current_tour.append(next_state)
        self.current_state = next_state
        self.done = True
        self.reset_tour()
        return next_state, reward, self.done, self.current_tour, self.all_tours

    def reset_possible_stops(self) -> object:
        """
        Resets the list of possible stops and loads all states expect the microhub.
        :return: None
        """
        copy_states = self.states.copy()
        copy_states.remove(self.microhub)
        self.current_state = self.microhub
        self.possible_stops = copy_states

    def reset_tours(self) -> object:
        """
        Resets tour and all tours, aswell as the tour meta data.
        :return: None
        """
        self.all_tours = []
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

    def reset_tour(self) -> object:
        """
        Resets tour and tour meta data. Adds the previous tour to all tours and adds the microhub as the first stop
        of the current tour.
        :return: None
        """
        self.all_tours.append(self.current_tour)
        self.current_tour = []
        self.current_tour.append(self.current_state)
        self.current_tour_weight = 0.0
        self.current_tour_volume = 0.0

    def update_tour_meta(self, next_state: object) -> object:
        self.possible_stops.remove(next_state)
        self.current_tour.append(next_state)
        self.current_tour_weight += next_state.demand_weight
        self.current_tour_volume += next_state.demand_volume

    def reward_func(self, current_stop: object, next_stop: object) -> object:
        """
        :param current_stop: current state in the environment
        :param next_stop: next possible stop
        :return: reward of traversing between the given stops
        """
        reward = self.distance_matrix.at[current_stop.hash_id, next_stop.hash_id]
        return reward

    def reward_func_hash(self, current_stop: float, next_stop: float) -> object:
        """
        :param current_stop: current state hash in the environment
        :param next_stop: next possible stop hash
        :return: reward of traversing between the given stops
        """
        return self.distance_matrix.at[current_stop, next_stop]

    def get_capacity_demand_of_stop(self, stop_hash: float) -> object:
        """
        :param stop_hash: hash_id of given stop
        :return: capacity demand of specified stop
        """
        return list(self.capacity_demands[stop_hash].values())

    def get_microhub(self) -> object:
        """
        :return: microhub as object
        """
        return self.microhub

    def get_microhub_hash(self) -> object:
        """
        :return: microhub hash_id
        """
        return self.microhub.hash_id

    def get_state_by_hash(self, hashIdentifier: float) -> object:
        """
        :param hashIdentifier: hash_id of given stop
        :return: stop by hash_id
        """
        return next((state for state in self.states if state.hash_id == hashIdentifier), None)

    def get_all_state_hashes(self) -> object:
        """
        :return: all state hash_ids
        """
        return [state.hash_id for state in self.states]

    def possible_rewards(self, state: object, action_space_list: object) -> object:
        """
        :param state: current state hash_id
        :param action_space_list: list of possible next states hash_ids
        :return: list of possible rewards
        """
        possible_rewards = self.distance_matrix.loc[state, action_space_list]
        return possible_rewards

    def get_next_legal_action(self) -> object:
        """
        :return: action, legal next states, legal next states with hub counter ignored, local search distances,
        bin packing capacities, microhub counter
        """
        legal_next_states = []
        legal_next_states_hubs_ignored = []
        legal_next_states_local_search_distance = dict()
        legal_next_states_bin_packing_capacities = dict()

        for stop in self.possible_stops:
            possible_tour_weight = float(stop.demand_weight) + self.current_tour_weight
            possible_tour_volume = float(stop.demand_volume) + self.current_tour_volume
            if possible_tour_weight <= self.vehicle_weight and possible_tour_volume <= self.vehicle_volume:
                if stop.hash_id == self.microhub.hash_id:
                    continue
                else:
                    legal_next_states.append(stop.hash_id)
                    legal_next_states_hubs_ignored.append(stop.hash_id)
                    legal_next_states_local_search_distance[stop.hash_id] = self.reward_func(
                        self.current_state, stop)
                    legal_next_states_bin_packing_capacities[stop.hash_id] = [
                        possible_tour_weight / self.vehicle_weight, possible_tour_volume / self.vehicle_volume]

        legal_next_states_local_search_distance = {k: v for k, v in
                                                   sorted(legal_next_states_local_search_distance.items(),
                                                          key=lambda x: x[1])}
        legal_next_states_bin_packing_capacities = {k: v for k, v in
                                                     sorted(legal_next_states_bin_packing_capacities.items(),
                                                            key=lambda x: x[1], reverse=True)}

        if legal_next_states:
            action = 1
            return action, legal_next_states, legal_next_states_hubs_ignored, legal_next_states_local_search_distance, legal_next_states_bin_packing_capacities, self.microhub_counter

        if not legal_next_states and not self.possible_stops:
            microhub_counter = self.microhub_counter + 1
            legal_next_states.append('{}/{}'.format(self.microhub.hash_id, microhub_counter))
            legal_next_states_hubs_ignored.append(self.microhub.hash_id)
            action = 2
            self.microhub_counter += 1
            return action, legal_next_states, legal_next_states_hubs_ignored, legal_next_states_local_search_distance, legal_next_states_bin_packing_capacities, self.microhub_counter

        if not legal_next_states and self.possible_stops:
            microhub_counter = self.microhub_counter + 1
            legal_next_states.append('{}/{}'.format(self.microhub.hash_id, microhub_counter))
            legal_next_states_hubs_ignored.append(self.microhub.hash_id)
            action = 0
            self.microhub_counter += 1
            return action, legal_next_states, legal_next_states_hubs_ignored, legal_next_states_local_search_distance, legal_next_states_bin_packing_capacities, self.microhub_counter
