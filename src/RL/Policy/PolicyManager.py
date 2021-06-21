import math
import random

import numpy as np
import pandas as pd

from src.Utils.helper import activationBySoftmax, normalize_list, softmaxDict
from src.Utils.memoryLoader import load_memory_df_from_local, save_memory_df_to_local
import keras.backend as K  # dont remove


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
                 baseline_theta):
        # --------------------
        # STATES / ACTIONS
        self.state_hashes = state_hashes
        self.microhub_hash = state_hashes[0]
        self.microhub_counter = 0

        # --------------------
        # MODEL-PARAMETER
        self.discountFactor = discount_factor
        self.learning_rate = learning_rate
        self.eps = exploration_rate

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

    def policy_update_by_learning(self, env, episode, episode_reward, gamma, max_steps, num_episodes, epoch):
        # --------------------
        # COPY OLD POLICY
        policy_action_space_copy = self.policy_action_space.copy()
        policy_action_stable = False
        policy_action_space_stable = False

        # --------------------
        # PREPARE EPISODE MEMORY
        state_memory = np.array([step_t.state for step_t in episode])
        action_memory = np.array([step_t.action for step_t in episode])
        action_space_memory = np.array([step_t.next_state for step_t in episode])
        reward_memory = np.array([step_t.reward for step_t in episode])

        # --------------------
        # CALCULATE G_t
        G_t = self.compute_G_t(reward_memory, gamma)
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

        if episode_reward < self.old_policy_reward:
            print(
                "----------------------------------------------GOOD EPISODE----------------------------------------------------")
            self.enhance_good_episode = True
        else:
            self.enhance_good_episode = False

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
            # earned_reward = episode[idx].reward
            softmax_weights = softmaxDict(self.policy_action_space.loc[state_hash, :])
            baseline_estimate = self.calculate_value_func(env, episode[idx],
                                                          softmax_weights, gamma,
                                                          episode[idx].microhub_counter)
            print("Relevant_Baseline_Estimate: ", baseline_estimate[episode[idx].state.stopid])

            # --------------------
            # ADVANTAGE / APPLY TEMPORAL DIFFERENCE ERROR
            # USES SIMPLE MONTE CARLO
            advantage_estimate = baseline_estimate[episode[idx].state.stopid] + self.learning_rate * (
                    g - baseline_estimate[episode[idx].state.stopid])
            # advantage_estimate = g - baseline_estimate[episode[idx].state.stopid]
            # advantage_estimate = g + ((self.discountFactor * baseline_estimate[episode[idx].next_state.stopid])-baseline_estimate[episode[idx].state.stopid])
            print("Current_g: ", g)
            # print("test ", test)
            print("Advantage_Estimate: ", advantage_estimate)

            # --------------------
            # CALCULATE LOSE/COST
            # lose = self.custom_loss(current_weight, g, advantage_estimate)
            # lose = self.cost(current_weight, g, advantage_estimate)
            # loseHistory.append(lose)
            # print("Current_lose: ", lose)
            # gradients = self.grad(current_weight, g, advantage_estimate)

            # --------------------
            # SETUP LEARNING RATE AND GAMMA_T
            lr = self.learning_rate
            print("Learning_rate: ", lr)

            # gamma_t = self.discountFactor/(1 + self.learning_rate_decay * epoch)
            gamma_t = self.discountFactor
            print("Gamma_t: ", gamma_t)

            # --------------------
            # CALCULATE AND UPDATE VALUE WEIGHT
            value_weight = softmax_weights.get(next_state_hash)
            value_weight = clip_weight(value_weight, 0.0001)
            print("Current_value_weight: ", value_weight)

            value_weight_new = value_weight + (lr * gamma_t * (baseline_estimate[episode[idx].next_state.stopid]))
            value_weight_new = clip_weight(value_weight_new, 0.0001)
            print("Current_value_weight_new: ", value_weight_new)

            # --------------------
            # DO STOCHASTIC GRADIENT STEP AND UPDATE PARAMETER OF POLICY
            gradient_step = lr * gamma_t * (advantage_estimate * np.log(value_weight_new))
            print("Gradient_step: ", gradient_step)

            # --------------------
            # APPLY MONTE-CARLO
            updated_weight = current_weight - gradient_step
            print("Current_weight: ", round(current_weight, 5))
            print("Pre_updated__weight: ", round(updated_weight, 5))

            # --------------------
            # APPLY PROBABILITY IN-/DECREASING FACTOR
            if updated_weight > current_weight:
                updated_weight = current_weight ** (self.increasing_factor_good_episode if self.enhance_good_episode else self.increasing_factor)
            if updated_weight < current_weight:
                updated_weight = current_weight ** (self.decreasing_factor_good_episode if self.enhance_good_episode else self.decreasing_factor)

            # --------------------
            # REDUCE ALTERNATIVE ACTION EVALUATIONS
            if self.enhance_good_episode:
                for state in episode[idx].possible_next_states:
                    if state != next_state_hash:
                        self.policy_action_space.at[state_hash, state] = self.policy_action_space.at[
                                                                         state_hash, state] ** (self.decreasing_factor_good_episode if self.enhance_good_episode else self.decreasing_factor)

            # --------------------
            # BACKPROPAGATION
            # Backpropagation mit Trägheitsterm fehlt
            # grad_weight = -self.learning_rate * (lose / current_weight)
            # weight_new = self.resolve_weight(grad_weight, current_weight, episode[idx].possible_rewards)

            # --------------------
            # SET UPDATED NEW WEIGHT
            print("Updated_weight: ", round(updated_weight, 5))
            self.policy_action_space.at[state_hash, next_state_hash] = round(updated_weight, 5)

        # --------------------
        # DECAY LEARNING RATE
        """
        self.learning_rate = self.learning_rate * (1 / (1 + self.learning_rate_decay * epoch))  # learning rate decay
        """

        # --------------------
        # BUILD AND COMPARE POLICIES (OLD vs. NEW)
        new_policy_reward, new_tour = self.construct_policy(self.policy_action_space, env, max_steps)
        print("-------------------Finalize-------------------")
        print("New_Policy_Reward: ", new_policy_reward)
        policy_relevant_reward = new_policy_reward

        # --------------------
        # EVALUATE INCREASING EPSILON
        # IF REWARD WAS STABLE OVER 3 TIMESTEPS, increase epsilon
        eps = self.eps
        if (self.penultimate_reward == self.old_policy_reward == new_policy_reward) and epoch < (0.7 * num_episodes):
            print("Increased exploration chance")
            eps = self.eps ** 0.2

        # --------------------
        # RESET PARAMETERS
        self.increasing_factor = 0.95
        self.decreasing_factor = 1.05
        self.enhance_good_episode = False
        # self.baseline_estimate = np.zeros_like(self.state_hashes, dtype=np.float32)
        self.penultimate_reward = self.old_policy_reward
        self.old_policy_reward = new_policy_reward

        # --------------------
        # RETURN
        return G_t, J_avR, loseHistory, eps, policy_relevant_reward

    def construct_policy(self, policy, env, max_steps):
        policy_reward = 0.0
        allTours = []
        tour = []
        state = env.reset()
        tour.append(state)
        for step_t in range(max_steps):
            legal_next_action, legal_next_states, legal_next_states_hubs_ignored, microhub_counter = env.getLegalAction()
            action_space = self.get_action_space_by_policy(state, legal_next_states, policy, microhub_counter)
            next_state, reward, done, currentTour, currentTours = env.step(legal_next_action, action_space)
            policy_reward += reward

            state = next_state
            tour.append(state)
            if legal_next_action == 0 or legal_next_action == 2:
                allTours.append(tour)
                tour = []
                tour.append(state)
            if done:
                break

        return policy_reward, allTours

    def handle_multiple_tours(self, state, next_state, microhub_counter):
        if state.hashIdentifier == self.microhub_hash:
            state_hash = '{}/{}'.format(state.hashIdentifier, microhub_counter)
        else:
            state_hash = state.hashIdentifier
        if next_state.hashIdentifier == self.microhub_hash:
            next_state_hash = '{}/{}'.format(next_state.hashIdentifier, microhub_counter)
        else:
            next_state_hash = next_state.hashIdentifier

        return state_hash, next_state_hash

    def calculate_value_func(self, env, episode, weights_dict, gamma, microhub_counter, theta=0.0001):
        while True:
            delta = 0.0
            old_values = np.copy(self.baseline_estimate)
            for s_a in self.state_hashes:
                s_a_stop = env.getStateByHash(s_a)
                v = np.zeros_like(self.state_hashes, dtype=np.float32)
                for s_n in self.state_hashes:
                    s_next = env.getStateByHash(s_n)
                    pi_s_a = 1
                    p = weights_dict.get(s_n if s_n != self.microhub_hash else '{}/{}'.format(s_n, microhub_counter))
                    r = env.reward_func_hash(s_a_stop.hashIdentifier, s_n)
                    # v[s_next.stopid] = r + gamma * (p * self.baseline_estimate[s_next.stopid])
                    v[s_next.stopid] = p * (r + gamma * self.baseline_estimate[s_next.stopid])
                    # if (s_next.hashIdentifier == self.microhub_hash):
                    #    v[s_next.stopid] *= self.microhub_counter
                self.baseline_estimate[s_a_stop.stopid] = np.sum(v)
                delta = np.maximum(delta,
                                   np.absolute(old_values[s_a_stop.stopid]) - self.baseline_estimate[s_a_stop.stopid])
            if delta < theta:
                break
        return self.baseline_estimate

    def resolve_weight(self, grad_weight, current_weight, possible_rewards):
        weight_new = current_weight
        if grad_weight == 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** self.increasing_factor  # try to increase probability by 5%
        if grad_weight > 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** self.increasing_factor
        if grad_weight < 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** self.decreasing_factor  # try to decrease probability by 5%
        return weight_new

    def sigmoidActivation(self, z):
        return 1 / (1 + np.exp(-z))

    def dotProduct(self, W, X):
        return self.sigmoidActivation(np.dot(X, W))

    def calculateCost(self, W, X, Y):
        y_pred = self.dotProduct(W, X)
        return -1 * (Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))

    def calculateGradient(self, W, X, Y):
        y_pred = self.dotProduct(W, X)
        A = (Y * (1 - y_pred) - (1 - Y) * y_pred)
        g = -1 * np.dot(A.T, X)
        return g

    def custom_loss(self, y_true, y_pred, advantages):  # objective function
        log_lik = y_true * K.log(y_pred)
        return K.mean(-log_lik * advantages)

    def get_action_space_by_policy(self, state, legal_next_states, policy, microhub_counter):
        if (state.hashIdentifier == self.microhub_hash):
            action_space_prob = policy.loc[
                '{}/{}'.format(state.hashIdentifier, microhub_counter), legal_next_states].to_numpy()
        else:
            action_space_prob = policy.loc[state.hashIdentifier, legal_next_states].to_numpy()
        highest_prob = max(action_space_prob)
        index_highest_prob = np.where(action_space_prob == highest_prob)[0]
        index = index_highest_prob[0]
        highest_prob_action_space = legal_next_states[index]
        return highest_prob_action_space

    def get_action_space(self, eps, state, legal_next_states, microhub_counter_env):
        # Check MicroHub counter
        if (self.microhub_counter != microhub_counter_env):
            self.microhub_counter = microhub_counter_env
            if ('{}/{}'.format(self.microhub_hash, self.microhub_counter) not in self.policy_action_space.index.values):
                new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
                self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
                self.policy_action_space['{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 0.0
                self.policy_action_space.fillna(value=0.05, inplace=True)
                # self.policy_action_space[self.policy_action_space.eq(0.0)] = self.policy_action_space + (1 / len(self.state_hashes))

        # epsilon greedy
        p = np.random.random()
        if p < eps:
            print("Choosed random action from action space.")
            highest_prob_action_space = random.choice(legal_next_states)
            return highest_prob_action_space, 1
        else:
            if (state.hashIdentifier == self.microhub_hash):
                action_space_prob = self.policy_action_space.loc[
                    '{}/{}'.format(state.hashIdentifier, self.microhub_counter), legal_next_states].to_numpy()
            else:
                action_space_prob = self.policy_action_space.loc[state.hashIdentifier, legal_next_states].to_numpy()
            # softmax_space_prob = activationBySoftmax(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            normalized_action_space_prob = normalize_list(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            # highest_prob_action_space_nr = np.random.choice(len(legal_next_states), p=softmax_space_prob)
            # highest_prob_action_space = legal_next_states[highest_prob_action_space_nr]
            # highest_prob = softmax_space_prob[highest_prob_action_space_nr]
            highest_prob = max(normalized_action_space_prob)
            index_highest_prob = np.where(normalized_action_space_prob == highest_prob)[0]
            if len(index_highest_prob) > 1:
                highest_prob_action_space = np.random.choice(legal_next_states, p=normalized_action_space_prob)
            else:
                highest_prob_action_space = legal_next_states[
                    index_highest_prob[0] if len(index_highest_prob) > 1 else 0]
            return highest_prob_action_space, highest_prob

    def get_action(self, state):
        action_prob = self.policy_action.loc[state.hashIdentifier].to_numpy()
        highest_prob_action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return highest_prob_action, action_prob[highest_prob_action]

    def get_current_policy(self):
        return self.policy_action_space

    def get_current_baseline_as_dict(self):
        baseline_dict = dict()
        for state_hash, baseline_value in zip(self.state_hashes, self.baseline_estimate):
            baseline_dict[state_hash] = baseline_value
        return baseline_dict

    def compute_G_t(self, reward_memory, gamma):
        """
        args
          a list of rewards
        returns
          a list of cummulated rewards G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + .. + gamma^{T-t-1}*R_{T}
        """
        G_t = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * self.discountFactor
                # discount *= gamma
            G_t[t] = G_sum
        return G_t

    def compute_J_avR(self, G_t):
        """
        calculate the objective function
        average reward per time-step
        purpose to measure the quality of a policy pi
        """
        return np.mean(G_t)

    def apply_aco_on_policy(self, increasing_factor, aco_probability_matrix):
        for index, row in aco_probability_matrix.iterrows():
            for state, aco_probability in row.items():
                if state not in self.policy_action_space.index.values:
                    new_row = pd.Series(name=state)
                    self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
                    self.policy_action_space[state] = 0.0
                    self.policy_action_space.fillna(value=0.0, inplace=True)
                if aco_probability > 0.00:
                    self.policy_action_space.at[index, state] = self.policy_action_space.at[index, state] ** increasing_factor

    def saveModel(self, model_name):
        save_memory_df_to_local('./model/' + model_name + '.pkl', self.policy_action_space)

    def loadModel(self, model_name):
        loaded_model = load_memory_df_from_local('./model/' + model_name + '.pkl', self.state_hashes, self.microhub_hash)
        self.policy_action_space = loaded_model
