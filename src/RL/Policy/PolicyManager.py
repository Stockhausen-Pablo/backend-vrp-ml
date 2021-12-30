import random

import numpy as np
import pandas as pd

from src.Utils.helper import normalize_list, action_by_softmax_as_dict
from src.Utils.memoryLoader import load_memory_df_from_local, save_memory_df_to_local


def clip_weight(current_weight, clipValue):
    if current_weight >= 1:
        current_weight = 1 - clipValue
    if current_weight <= 0:
        current_weight = 0 + clipValue
    return current_weight


class PolicyManager:
    def __init__(self,
                 state_hashes,
                 learning_rate,
                 discount_factor,
                 exploration_rate,
                 increasing_factor,
                 increasing_factor_good_episode,
                 decreasing_factor,
                 decreasing_factor_good_episode,
                 baseline_theta,
                 distance_utilization_threshold,
                 capacity_utilization_threshold,
                 local_search_threshold,
                 policy_reset_threshold):

        # --------------------
        # STATES / ACTIONS
        self.state_hashes = state_hashes
        self.microhub_hash = state_hashes[0]
        self.microhub_counter = 0

        # --------------------
        # MODEL-PARAMETER
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.eps = exploration_rate
        self.distance_utilization_threshold = distance_utilization_threshold
        self.capacity_utilization_threshold = capacity_utilization_threshold
        self.local_search_threshold = local_search_threshold
        self.policy_reset_threshold = policy_reset_threshold

        # --------------------
        # CONTROL-PARAMETER
        self.learning_rate_decay = 0.1
        self.increasing_factor = increasing_factor
        self.decreasing_factor = decreasing_factor
        self.increasing_factor_good_episode = increasing_factor_good_episode
        self.decreasing_factor_good_episode = decreasing_factor_good_episode
        self.enhance_good_episode = False
        self.theta = float(baseline_theta)
        self.penultimate_reward = 0.0

        # --------------------
        # MODEL-CONTEXT SPECIFIC
        self.G = 0
        self.old_policy_reward = 0.0
        self.baseline_estimate = np.zeros_like(state_hashes, dtype=np.float32)

        # --------------------
        # PARAMETERIZED POLICY
        self.policy_action_space = pd.DataFrame()

    def policy_update_by_learning(self, env: object, episode: object, episode_reward: int, gamma: float, max_steps: int, num_episodes: int, epoch: int) -> object:
        """
        Takes the episodes "on-policy" experience and updates the policy.
        :param env: environment instance
        :param episode: List of every taken step in the episode
        :param episode_reward: Cumulative reward of the episode
        :param gamma: gamma factor
        :param max_steps: max steps amount that the policy manager is allowed to use
        :param num_episodes: the overall amount of defined episodes
        :param epoch: current epoch
        :return: Cumulative discount reward, average reward, lose history, epsilon, current policy reward
        """
        # --------------------
        # COPY OLD POLICY
        policy_action_space_copy = self.policy_action_space.copy()

        # --------------------
        # PREPARE EPISODE MEMORY
        # state_memory = np.array([step_t.state for step_t in episode])
        # action_memory = np.array([step_t.action for step_t in episode])
        # action_space_memory = np.array([step_t.next_state for step_t in episode])
        reward_memory = np.array([step_t.reward for step_t in episode])

        # --------------------
        # CALCULATE G_t
        G_t = self.compute_G_t(reward_memory)
        loseHistory = []
        J_avR = self.compute_J_avR(G_t)
        std_deviation = np.std(G_t) if np.std(G_t) > 0 else 1
        self.G = (G_t - J_avR) / std_deviation

        # --------------------
        # COMPARE OLD POLICY REWARD TO EPISODE REWARD
        # increase probability factor when current episode reward > old policy reward
        print("------------------Comparison------------------")
        print("--------Old_Policy-vs-Current_Episode---------")
        print("Epoch: ", epoch)
        print("Old_Policy_Reward: ", self.old_policy_reward)
        print("Current_Episode_Reward: ", episode_reward)
        reward_difference = self.old_policy_reward - episode_reward

        if episode_reward < self.old_policy_reward:
            print(
                "----------------------------------------------GOOD EPISODE----------------------------------------------------")
            self.enhance_good_episode = True
        else:
            self.enhance_good_episode = False

        print("Enhance good episode: ", self.enhance_good_episode)

        for idx, g in enumerate(self.G):
            print("------------------Step ", idx, "------------------")
            # --------------------
            # HANDLE MICROHUB PROBLEMATIC
            state_hash, next_state_hash = self.handle_multiple_tours(episode[idx].state, episode[idx].next_state,
                                                                     episode[idx].microhub_counter)
            print("Current_state: ", state_hash)
            print("Next_state: ", next_state_hash)
            # --------------------
            # FIND CURRENT WEIGHT FOR ACTION
            current_weight = self.policy_action_space.at[state_hash, next_state_hash]
            current_weight = clip_weight(current_weight, 0.0001)  # to avoid zero division

            # --------------------
            # VALUE FUNCTION (Policy Evaluation)
            # Get possible lowest reward | goal to minimize reward (as lowest distance)
            softmax_weights = action_by_softmax_as_dict(self.policy_action_space.loc[state_hash, :])
            baseline_estimate = self.calculate_value_func(env,
                                                          softmax_weights,
                                                          gamma,
                                                          episode[idx].microhub_counter)
            print("Relevant_Baseline_Estimate: ", baseline_estimate[episode[idx].state.stop_id])

            # --------------------
            # ADVANTAGE / APPLY TEMPORAL DIFFERENCE ERROR
            # USES SIMPLE MONTE CARLO
            advantage_estimate = baseline_estimate[episode[idx].state.stop_id] + self.learning_rate * (
                    g - baseline_estimate[episode[idx].state.stop_id])

            print("Current_g: ", g)
            print("Advantage_Estimate: ", advantage_estimate)

            # --------------------
            # CALCULATE LOSE/COST
            lose = self.calculate_cost(current_weight, g, advantage_estimate)
            loseHistory.append(lose)
            print("Current_lose: ", lose)

            # --------------------
            # SETUP LEARNING RATE AND GAMMA_T
            lr = self.learning_rate
            print("Learning_rate: ", lr)

            # gamma_t = self.discountFactor/(1 + self.learning_rate_decay * epoch)
            gamma_t = self.discount_factor
            print("Gamma_t: ", gamma_t)

            # --------------------
            # CALCULATE AND UPDATE VALUE WEIGHT
            value_weight = softmax_weights.get(next_state_hash)
            value_weight = clip_weight(value_weight, 0.0001)
            print("Current_value_weight: ", value_weight)

            value_weight_new = value_weight + (lr * gamma_t * (baseline_estimate[episode[idx].next_state.stop_id]))
            value_weight_new = clip_weight(value_weight_new, 0.0001)
            print("Current_value_weight_new: ", value_weight_new)

            # --------------------
            # DO STOCHASTIC GRADIENT STEP AND UPDATE PARAMETER OF POLICY
            gradient_step = lr * gamma_t * (advantage_estimate * np.log(value_weight_new))
            print("Gradient_step: ", gradient_step)

            # --------------------
            # APPLY MONTE-CARLO
            updated_weight = current_weight - gradient_step
            print("Current_weight: ", current_weight)
            print("Pre_updated__weight: ", updated_weight)

            # --------------------
            # APPLY PROBABILITY IN-/DECREASING FACTOR
            final_weight = updated_weight
            if updated_weight > current_weight:
                final_weight = current_weight ** (
                    self.increasing_factor_good_episode if self.enhance_good_episode is True else self.increasing_factor)
            if updated_weight < current_weight and self.enhance_good_episode is False:
                final_weight = current_weight ** (
                    self.decreasing_factor_good_episode if self.enhance_good_episode is True else self.decreasing_factor)

            # --------------------
            # REDUCE ALTERNATIVE ACTION EVALUATIONS
            if self.enhance_good_episode is True and updated_weight > current_weight:
                reward_difference_reduced = (
                        ((10 * np.log10(reward_difference)) / np.log(10)) / 100) if reward_difference > 0 else 0
                to_update_states = episode[idx].possible_next_states
                to_update_states.remove(next_state_hash)
                self.policy_action_space.loc[state_hash, to_update_states] **= (
                        self.decreasing_factor_good_episode + (reward_difference_reduced if reward_difference_reduced > 0 else 0))

            # --------------------
            # SET UPDATED NEW WEIGHT
            print("Final_weight: ", final_weight)
            self.policy_action_space.at[state_hash, next_state_hash] = final_weight

        # --------------------
        # DECAY LEARNING RATE
        """
        self.learning_rate = self.learning_rate * (1 / (1 + self.learning_rate_decay * epoch))  # learning rate decay
        """

        # --------------------
        # BUILD AND COMPARE POLICIES (OLD vs. NEW)
        print('Constructing new Policy')
        self.microhub_counter = 0
        new_policy_reward, new_tour = self.construct_policy(self.policy_action_space, env, max_steps)
        print("-------------------Finalize-------------------")
        print("Enhance good episode: ", self.enhance_good_episode)
        print("Old_Policy_Reward: ", self.old_policy_reward)
        print("New_Policy_Reward: ", new_policy_reward)
        policy_relevant_reward = new_policy_reward

        if self.enhance_good_episode is False and ((self.old_policy_reward - new_policy_reward) < self.policy_reset_threshold) and self.old_policy_reward > 0.0:
            print("-Resetting policy to old standard-")
            self.policy_action_space = policy_action_space_copy
            policy_relevant_reward = self.old_policy_reward

        if self.enhance_good_episode is True and ((self.old_policy_reward - new_policy_reward) < -5) and self.old_policy_reward > 0.0:
            print("-Resetting policy to old standard-")
            self.policy_action_space = policy_action_space_copy
            policy_relevant_reward = self.old_policy_reward

        # --------------------
        # EVALUATE INCREASING EPSILON
        # IF REWARD WAS STABLE OVER 3 TIMESTEPS, increase epsilon
        eps = self.eps
        if (self.penultimate_reward == self.old_policy_reward == new_policy_reward) and epoch < (0.7 * num_episodes):
            print("Increased exploration chance")
            eps = self.eps ** 0.8

        # --------------------
        # RESET PARAMETERS
        self.microhub_counter = 0
        self.enhance_good_episode = False
        self.baseline_estimate = np.zeros_like(self.state_hashes, dtype=np.float32)
        self.penultimate_reward = self.old_policy_reward
        self.old_policy_reward = new_policy_reward

        # --------------------
        # RETURN
        return G_t, J_avR, loseHistory, eps, policy_relevant_reward, new_tour

    def construct_policy(self, policy: object, env: object, max_steps: int) -> object:
        """
        Construct current policy by optimal path.
        :param policy: current policy
        :param env: environment instance
        :param max_steps: max steps amount that the policy manager is allowed to use
        :return: policy reward, all constructed tours
        """
        policy_reward = 0.0
        all_tours = []
        tour = []
        state = env.reset()
        tour.append(state)

        for step_t in range(max_steps):
            print("constructing policy: get legal next states")
            legal_next_action, legal_next_states, legal_next_states_hubs_ignored, legal_next_states_local_search_distance, legal_next_states_bin_packing_capacities, microhub_counter = env.get_next_legal_action()
            print("constructing policy: getting action space")
            action_space = self.get_action_space_by_policy(state, legal_next_states, policy, microhub_counter)
            print("constructing policy: doing step")
            next_state, reward, done, current_tour, current_tours = env.step(legal_next_action, action_space)
            policy_reward += reward

            state = next_state
            tour.append(state)
            if legal_next_action == 0 or legal_next_action == 2:
                all_tours.append(tour)
                tour = [state]
            if done:
                break

        return policy_reward, all_tours

    def handle_multiple_tours(self, state: object, next_state: object, microhub_counter: int) -> object:
        """
        Defines handle logic for multiple hub entries in policy.
        :param state: current state
        :param next_state: next state
        :param microhub_counter: counter number specifying how often the tour constructor returned to the hub
        :return: state hash_id, next state hash_id
        """
        if state.hash_id == self.microhub_hash:
            state_hash = '{}/{}'.format(state.hash_id, microhub_counter)
        else:
            state_hash = state.hash_id
        if next_state.hash_id == self.microhub_hash:
            next_state_hash = '{}/{}'.format(next_state.hash_id, microhub_counter)
        else:
            next_state_hash = next_state.hash_id

        return state_hash, next_state_hash

    def calculate_value_func(self, env: object, weights_dict: dict, gamma: float, microhub_counter: int, theta: float = 0.0001) -> object:
        """
        Calculates state-value function.
        :param env: environment instance
        :param weights_dict: dictionary of states with associated softmax weights
        :param gamma: gamma factor
        :param microhub_counter: counter number specifying how often the tour constructor returned to the hub
        :param theta: threshold indicator for termination
        :return: baseline estimate
        """
        while True:
            delta = 0.0
            old_values = np.copy(self.baseline_estimate)
            for s_a in self.state_hashes:
                s_a_stop = env.get_state_by_hash(s_a)
                v = np.zeros_like(self.state_hashes, dtype=np.float32)
                for s_n in self.state_hashes:
                    s_next = env.get_state_by_hash(s_n)
                    pi_s_a = 1
                    p = weights_dict.get(s_n if s_n != self.microhub_hash else '{}/{}'.format(s_n, microhub_counter))
                    r = env.reward_func_hash(s_a_stop.hash_id, s_n)
                    # v[s_next.stopid] = r + gamma * (p * self.baseline_estimate[s_next.stopid])
                    v[s_next.stop_id] = pi_s_a * p * (r + gamma * self.baseline_estimate[s_next.stop_id])
                    # if (s_next.hashIdentifier == self.microhub_hash):
                    #    v[s_next.stopid] *= self.microhub_counter
                self.baseline_estimate[s_a_stop.stop_id] = np.sum(v)
                delta = np.maximum(delta,
                                   np.absolute(old_values[s_a_stop.stop_id]) - self.baseline_estimate[s_a_stop.stop_id])
            if delta < theta:
                break
        return self.baseline_estimate

    def resolve_weight(self, grad_weight: object, current_weight: float, possible_rewards: object) -> object:
        """
        Clipping
        :return: clipped weight
        """
        weight_new = current_weight
        if grad_weight == 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** self.increasing_factor  # try to increase probability by 5%
        if grad_weight > 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** self.increasing_factor
        if grad_weight < 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** self.decreasing_factor  # try to decrease probability by 5%
        return weight_new

    @staticmethod
    def sigmoid_activation(z: object) -> object:
        return 1 / (1 + np.exp(-z))

    def calculate_dot_product(self, W: object, X: object) -> object:
        return self.sigmoid_activation(np.dot(X, W))

    def calculate_cost(self, W: object, X: object, Y: object) -> object:
        """
        Calculate the cost for choosing an specific action.
        :param W: the current weight of the choosen action.
        :param X: Discounted reward for current timestep.
        :param Y: Advantage estimate
        :return: cost
        """
        y_pred = self.calculate_dot_product(W, X)
        return -1 * (Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))

    def get_action_space_by_policy(self, state: object, legal_next_states: object, policy: object, microhub_counter: int) -> object:
        """
        Gets the action space following a specific given policy.
        :param state: current state
        :param legal_next_states: possible next states
        :param policy: current policy
        :param microhub_counter: counter number specifying how often the tour constructor returned to the hub
        :return:
        """
        if state.hash_id == self.microhub_hash:
            self.policy_action_space.fillna(value=0.05, inplace=True)
            if '{}/{}'.format(self.microhub_hash, microhub_counter) not in self.policy_action_space.index.values:
                new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, microhub_counter))
                self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
                self.policy_action_space.fillna(value=0.05, inplace=True)
                self.policy_action_space['{}/{}'.format(self.microhub_hash, microhub_counter)] = 1 / len(self.state_hashes)
            action_space_prob = policy.loc['{}/{}'.format(state.hash_id, microhub_counter), legal_next_states].to_numpy()
        else:
            if len(legal_next_states) == 1:
                if legal_next_states[0] not in self.policy_action_space.index.values:
                    new_row = pd.Series(name=legal_next_states[0])
                    self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
                    self.policy_action_space.fillna(value=0.05, inplace=True)
                    self.policy_action_space[legal_next_states[0]] = 1 / len(self.state_hashes)
            action_space_prob = policy.loc[state.hash_id, legal_next_states].to_numpy()
        highest_prob = max(action_space_prob)
        index_highest_prob = np.where(action_space_prob == highest_prob)[0]
        index = index_highest_prob[0]
        highest_prob_action_space = legal_next_states[index]
        return highest_prob_action_space

    def get_action_space(self, eps: object, state: object, legal_next_states: object, legal_next_states_local_search_distance: object,
                         legal_next_states_bin_packing_capacities: object,
                         microhub_counter_env: object) -> object:
        """
        Gets the action space following the current policy.
        :param eps: exploration threshold
        :param state: current state
        :param legal_next_states: next possible states
        :param legal_next_states_local_search_distance: next possible states under perspective of the distance
        :param legal_next_states_bin_packing_capacities: next possible states under perspective of the capacities
        :param microhub_counter_env: Given microhunter by the environment.
        :return: next action, next action probability
        """
        if self.microhub_counter != microhub_counter_env:
            self.microhub_counter = microhub_counter_env
            if ('{}/{}'.format(self.microhub_hash, self.microhub_counter) not in self.policy_action_space.index.values):
                new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
                self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
                self.policy_action_space.fillna(value=0.05, inplace=True)
                self.policy_action_space['{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 1 / len(
                    self.state_hashes)
                # self.policy_action_space.fillna(value=0.05, inplace=True)
                # self.policy_action_space[self.policy_action_space.eq(0.0)] = self.policy_action_space + (1 / len(self.state_hashes))

        # epsilon greedy
        p = np.random.random()
        if p < eps:
            print("Choosed random action from action space.")
            highest_prob_action_space = random.choice(legal_next_states)
            return highest_prob_action_space, 1
        else:
            if state.hash_id == self.microhub_hash:
                action_space_prob = self.policy_action_space.loc[
                    '{}/{}'.format(state.hash_id, self.microhub_counter), legal_next_states].to_numpy()
            else:
                action_space_prob = self.policy_action_space.loc[state.hash_id, legal_next_states].to_numpy()
            # softmax_space_prob = activationBySoftmax(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            normalized_action_space_prob = normalize_list(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            # highest_prob_action_space_nr = np.random.choice(len(legal_next_states), p=softmax_space_prob)
            # highest_prob_action_space = legal_next_states[highest_prob_action_space_nr]
            # highest_prob = softmax_space_prob[highest_prob_action_space_nr]
            highest_prob = max(normalized_action_space_prob)
            index_highest_prob = np.where(normalized_action_space_prob == highest_prob)[0]
            if len(index_highest_prob) > 1:
                random_index = random.randint(0, len(index_highest_prob) - 1)
                highest_prob_action_space = legal_next_states[random_index]
                # highest_prob_action_space = np.random.choice(legal_next_states, p=normalized_action_space_prob)
            else:
                if highest_prob == 1.0:
                    highest_prob_action_space = legal_next_states[0]
                else:
                    highest_prob_action_space = legal_next_states[index_highest_prob[0]]

            # --------------------
            # APPLY LOCAL SEARCH AND BIN-PACKING(First Fit Decreasing)
            if len(legal_next_states) > 1:
                lowest_state_distance = next(iter(legal_next_states_local_search_distance))
                highest_state_capacities_utilization = next(iter(legal_next_states_bin_packing_capacities))
                highest_state_capacities_utilization_weight = legal_next_states_bin_packing_capacities[highest_state_capacities_utilization][0]
                highest_state_capacities_utilization_distance = legal_next_states_local_search_distance[highest_state_capacities_utilization]

                lowest_distance = legal_next_states_local_search_distance[lowest_state_distance]
                lowest_distance_weight_utilization = legal_next_states_bin_packing_capacities[lowest_state_distance][0]

                choosen_distance = legal_next_states_local_search_distance[highest_prob_action_space]

                diviation_distance_highest_utilization = 1 - (lowest_distance / highest_state_capacities_utilization_distance)
                diviation_distance_choosen = 1 - (lowest_distance / choosen_distance)

                diff_diviation = diviation_distance_highest_utilization - diviation_distance_choosen

                if diff_diviation < self.distance_utilization_threshold:
                    highest_prob_action_space = highest_state_capacities_utilization

                if diviation_distance_highest_utilization > self.local_search_threshold and diviation_distance_choosen > self.local_search_threshold:
                    divation_weight_to_max = lowest_distance_weight_utilization / highest_state_capacities_utilization_weight
                    if divation_weight_to_max > self.capacity_utilization_threshold:
                        highest_prob_action_space = lowest_state_distance

                return highest_prob_action_space, highest_prob

            else:
                return highest_prob_action_space, highest_prob

    def get_action(self, state: object) -> object:
        """
        :param state: given state
        :return: most likely next state, highest probability
        """
        action_prob = self.policy_action.loc[state.hash_id].to_numpy()
        highest_prob_action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return highest_prob_action, action_prob[highest_prob_action]

    def get_current_policy(self) -> object:
        """
        :return: current policy
        """
        return self.policy_action_space

    def set_current_policy(self, policy) -> object:
        """
        :return: current policy
        """
        self.policy_action_space = policy
        return self.policy_action_space

    def get_current_baseline_as_dict(self) -> object:
        """
        :return: baseline estimate as dict
        """
        baseline_dict = dict()
        for state_hash, baseline_value in zip(self.state_hashes, self.baseline_estimate):
            baseline_dict[state_hash] = baseline_value
        return baseline_dict

    def compute_G_t(self, reward_memory: object) -> object:
        """
        :param reward_memory: list of rewards
        :returns: a list of cummulated rewards G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + .. + gamma^{T-t-1}*R_{T}
        """
        G_t = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * self.discount_factor
                # discount *= gamma
            G_t[t] = G_sum
        return G_t

    @staticmethod
    def compute_J_avR(G_t: object) -> object:
        """
        calculate the objective function
        purpose to measure the quality of a policy pi
        :returns: average reward per time-step
        """
        return np.mean(G_t)

    def apply_aco_on_policy(self, increasing_factor: float, aco_probability_matrix: object) -> object:
        """
        Applies the aco result on current policy.
        :param increasing_factor: increasing factor of aco
        :param aco_probability_matrix: aco result
        :return: None
        """
        for index, row in aco_probability_matrix.iterrows():
            for state, aco_probability in row.items():
                if state not in self.policy_action_space.index.values:
                    new_row = pd.Series(name=state)
                    self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
                    self.policy_action_space[state] = 1 / len(self.state_hashes)
                    self.policy_action_space.fillna(value=1 / len(self.state_hashes), inplace=True)
                if aco_probability > 0.00:
                    self.policy_action_space.at[index, state] = self.policy_action_space.at[
                                                                    index, state] ** increasing_factor

    def saveModel(self, model_name: str) -> object:
        """
        :param model_name: ML-Model name that will be saved
        :return: None
        """
        save_memory_df_to_local('./model/' + model_name + '.pkl', self.policy_action_space)

    def loadModel(self, model_name: str) -> object:
        """
        :param model_name: ML-Model name that will be loaded
        :return: None
        """
        loaded_model = load_memory_df_from_local('./model/' + model_name + '.pkl', self.state_hashes,
                                                 self.microhub_hash)
        self.policy_action_space = loaded_model
