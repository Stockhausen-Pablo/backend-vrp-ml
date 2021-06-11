import math
import random

import numpy as np
import pandas as pd

from src.Utils.helper import normalize_list, activationBySoftmax
import keras.backend as K


def clip_weight(current_weight, clipValue):
    if current_weight == 1:
        current_weight = current_weight - clipValue
    if current_weight == 0:
        current_weight = current_weight + clipValue
    return current_weight


class PolicyManager:
    def __init__(self, state_hashes, actions, aco_boost_probability_Matrix, discountFactor=1.0, theta=0.00001):
        self.state_hashes = state_hashes
        self.discountFactor = discountFactor
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.5
        self.eps = 0.05
        self.increasing_factor = 0.95
        self.decreasing_factor = 1.05
        self.decrease_overall = False
        self.theta = float(theta)
        self.penultimate_reward = 0.0
        self.microhub_hash = state_hashes[0]
        self.G = 0  # advantanges
        self.microhub_counter = 0
        self.baseline_estimate = np.zeros_like(state_hashes, dtype=np.float32)
        # Initialize a random start policy
        # shows the distribution of a state over the action space per row
        # self.policy_action_space = aco_boost_probability_Matrix
        self.policy_action_space = pd.DataFrame(index=state_hashes[1:], columns=state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
        self.policy_action_space = self.policy_action_space.append(new_row, ignore_index=False)
        self.policy_action_space['{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 0.0
        self.policy_action_space.fillna(value=0.0, inplace=True)
        self.policy_action_space = self.policy_action_space + (1 / len(state_hashes))
        self.policy_action = pd.DataFrame(index=state_hashes, columns=[actions])
        self.policy_action.fillna(value=0.0, inplace=True)
        self.policy_action = self.policy_action + (1 / len(actions))

    def policy_eval(self, env, policy, values, gamma=0.9, theta=0.01):
        microhubCounter = 0
        microhubId = env.getMicrohubId()
        while True:
            delta = 0.0
            old_values = np.copy(values)
            for s in env.states:
                v = np.zeros_like(env.states, dtype=np.float32)
                if (s.hashIdentifier == microhubId):
                    possibleStates = policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "possible_actions"]
                else:
                    possibleStates = policy.at[s.hashIdentifier, "possible_actions"]
                for s_n in possibleStates:
                    s_next = env.getStateByHash(s_n)
                    pi_s_a = 1
                    p = env.getTransitionProbability(s.hashIdentifier, s_next.hashIdentifier)
                    r = env.reward_func(s, s_next)
                    v[s_next.stopid] = pi_s_a * p * (r + gamma * values[s_next.stopid])

                values[s.stopid] = np.sum(v)
                delta = np.maximum(delta, np.absolute(values[s.stopid] - old_values[s.stopid]))

            if delta < theta:
                break
        return values

    def policy_improvement(self, q_values, episode):
        for i, step_t in enumerate(episode):
            argmax_q = -np.inf
            best_a = None
            q_actual = 0
            s_q_values = self.get_qvalue(step_t.state.hashIdentifier, step_t.action)
            test = s_q_values.index(max(s_q_values))
            for s_n in step_t.possible_next_states:
                test = 0

    # def policy_improvement(self, values, env, policy, episode, gamma, discount_factor=1.0):
    #     policy_stable = True
    #     for s in env.states:
    #         argmax_q = -np.inf
    #         best_a = None
    #         q_actual = 0
    #         possibleStates = []
    #         if (s.hashIdentifier == microhubId):
    #             possibleStates = policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "possible_actions"]
    #         else:
    #             possibleStates = policy.at[s.hashIdentifier, "possible_actions"]
    #         for s_n in possibleStates:
    #             s_next = env.getStateByHash(s_n)
    #             p = env.getTransitionProbability(s.hashIdentifier, s_next.hashIdentifier)
    #             r = env.reward_func(s, s_next)
    #             q_actual += p * (r + gamma * values[s_next.stopid])
    #
    #             if q_actual > argmax_q:
    #                 argmax_q = q_actual
    #                 best_a = s_next.hashIdentifier
    #
    #         if s.hashIdentifier == microhubId:
    #             if policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "best_a"] != best_a:
    #                 policy_stable = False
    #             policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "best_a"] = best_a
    #             microhubCounter += 1
    #         else:
    #             if policy.at[s.hashIdentifier, "best_a"] != best_a:
    #                 policy_stable = False
    #             policy.at[s.hashIdentifier, "best_a"] = best_a
    #
    #     return policy_stable

    # while True:
    #     V = self.policy_eval(env, episode, numIterations=20)
    #     policy_stable = True
    #     for s in state_hashes:
    #         chosen_a = policy[s].idxmax()
    #         action_values = one_step_lookahead(s, V)
    #         best_a = max(action_values, key=lambda key: action_values[key])
    #         if chosen_a != best_a:
    #             policy_stable = False
    #         policy[s] = np.eye(env.nA)[best_a]
    #     if policy_stable:
    #         return policy, V

    def policy_update_by_learning(self, env, episode, episode_reward, gamma, max_steps, num_episodes, epoch):
        # --------------------
        # COPY OLD POLICY
        policy_action_next = self.policy_action.copy()
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
        old_policy_reward, old_tour = self.construct_policy(policy_action_space_copy, env, max_steps)
        if episode_reward < old_policy_reward:
            self.increasing_factor = 0.92
            self.decreasing_factor = 1.07
            self.decrease_overall = True

        for idx, g in enumerate(self.G):
            # --------------------
            # HANDLE MICROHUB PROBLEMATIC
            state_hash, next_state_hash = self.handle_multiple_tours(episode[idx].state, episode[idx].next_state,
                                                                     episode[idx].microhub_counter)
            # --------------------
            # FIND CURRENT WEIGHT FOR ACTION
            current_weight = self.policy_action_space.at[state_hash, next_state_hash]
            current_weight = clip_weight(current_weight, 0.0001)  # to avoid zero division

            # --------------------
            # VALUE FUNCTION (Policy Evaluation)
            # Get possible lowest reward | goal to minimize reward (as lowest distance)
            # earned_reward = episode[idx].reward
            baseline_estimate = self.calculate_value_func(env, episode[idx],
                                                          self.policy_action_space.loc[state_hash, :], gamma,
                                                          episode[idx].microhub_counter)
            #q_value = baseline_estimate[episode[idx].next_state.stopid] + self.learning_rate * abs(episode[idx].reward + gamma * baseline_estimate[episode[idx+1 if idx < (len(self.G)-1) else idx].state.stopid] - baseline_estimate[episode[idx].next_state.stopid])

            # --------------------
            # ADVANTAGE
            advantage_estimate = G_t[idx] - baseline_estimate[episode[idx].next_state.stopid]
            # advantage_estimate = G_t[idx] - baseline

            # --------------------
            # CALCULATE LOSE/COST
            lose = self.custom_loss(current_weight, g, advantage_estimate)
            loseHistory.append(lose)

            # --------------------
            # BACKPROPAGATION
            # Backpropagation mit Trägheitsterm fehlt
            old_policy_prob = policy_action_space_copy.at[state_hash, next_state_hash]
            grad_weight = -self.learning_rate * (lose / current_weight)
            weight_new = self.resolve_weight(grad_weight, current_weight, episode[idx].possible_rewards)
            test = (grad_weight / old_policy_prob) * advantage_estimate

            # --------------------
            # SET UPDATED NEW WEIGHT
            if weight_new:
                if self.decrease_overall and weight_new > current_weight:
                    for state in episode[idx].possible_next_states:
                        self.policy_action_space.at[state_hash, state] = self.policy_action_space.at[state_hash, state] ** self.decreasing_factor
                self.policy_action_space.at[state_hash, next_state_hash] = weight_new

        # --------------------
        # DECAY LEARNING RATE
        """
        self.learning_rate = self.learning_rate * (1 / (1 + self.learning_rate_decay * epoch))  # learning rate decay
        """

        # --------------------
        # BUILD AND COMPARE POLICIES (OLD vs. NEW)
        new_policy_reward, new_tour = self.construct_policy(self.policy_action_space, env, max_steps)
        policy_relevant_reward = new_policy_reward
        if old_policy_reward < new_policy_reward and old_policy_reward < episode_reward:
            self.policy_action_space = policy_action_space_copy
            policy_relevant_reward = old_policy_reward

        # --------------------
        # EVALUATE INCREASING EPSILON
        # IF REWARD WAS STABLE OVER 3 TIMESTEPS, increase epsilon
        eps = self.eps
        if (self.penultimate_reward == old_policy_reward == new_policy_reward) and epoch < (0.7 * num_episodes):
            print("Increased exploration chance")
            eps = self.eps ** 0.5

        # --------------------
        # RESET PARAMETERS
        self.microhub_counter = 0
        self.increasing_factor = 0.95
        self.decreasing_factor = 1.05
        self.decrease_overall = False
        self.baseline_estimate = np.zeros_like(self.state_hashes, dtype=np.float32)
        self.penultimate_reward = old_policy_reward

        # --------------------
        # RETURN
        return G_t, J_avR, loseHistory, eps, policy_relevant_reward

    def construct_policy(self, policy, env, max_steps):
        policy_reward = 0.0
        tour = []
        state = env.reset()
        tour.append(state)
        for step_t in range(max_steps):
            legal_next_action, legal_next_states, legal_next_states_hubs_ignored, microhub_counter = env.getLegalAction()
            action_space = self.get_action_space_by_policy(state, legal_next_states, policy, microhub_counter)
            next_state, reward, done, currentTour, currentTours = env.step(legal_next_action, action_space)
            policy_reward += reward
            if done:
                break

            state = next_state
            tour.append(state)

        return policy_reward, tour

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

    def calculate_value_func(self, env, episode, weights, gamma, microhub_counter, theta=0.01):
        weights_dict = weights.to_dict()
        while True:
            delta = 0.0
            old_values = np.copy(self.baseline_estimate)
            for s_a in episode.possible_next_states_hub_ignored:
                s_a_stop = env.getStateByHash(s_a)
                v = np.zeros_like(self.state_hashes, dtype=np.float32)
                for s_n in episode.possible_next_states_hub_ignored:
                    s_next = env.getStateByHash(s_n)
                    pi_s_a = 1
                    p = weights_dict.get(s_n if s_n != self.microhub_hash else '{}/{}'.format(s_n, microhub_counter))
                    r = env.reward_func_hash(s_a_stop.hashIdentifier, s_n)
                    v[s_next.stopid] = pi_s_a * p * (r + gamma * self.baseline_estimate[s_next.stopid])
                self.baseline_estimate[s_a_stop.stopid] = np.sum(v)
                delta = np.maximum(delta, np.absolute(old_values[s_a_stop.stopid]) - self.baseline_estimate[s_a_stop.stopid])
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

    def custom_loss(self, y_true, y_pred, advantages):  # objective function
        log_lik = y_true * K.log(y_pred)
        return K.mean(-log_lik * advantages)

    def get_action_space_by_policy(self, state, legal_next_states, policy, microhub_counter):
        if (state.hashIdentifier == self.microhub_hash):
            action_space_prob = policy.loc[
                '{}/{}'.format(state.hashIdentifier, microhub_counter), legal_next_states].to_numpy()
        else:
            action_space_prob = policy.loc[state.hashIdentifier, legal_next_states].to_numpy()
        normalized_action_space_prob = normalize_list(action_space_prob) if len(legal_next_states) > 1 else [1.0]
        highest_prob = max(normalized_action_space_prob)
        index_highest_prob = np.where(normalized_action_space_prob == highest_prob)[0]
        if len(index_highest_prob) > 1:
            highest_prob_action_space = np.random.choice(legal_next_states, p=normalized_action_space_prob)
        else:
            highest_prob_action_space = legal_next_states[index_highest_prob[0] if len(index_highest_prob) > 1 else 0]
        # softmax_space_prob2 = activationBySoftmax(action_space_prob) if len(legal_next_states) > 1 else [1.0]
        # highest_prob_action_space = np.random.choice(legal_next_states, p=normalized_action_space_prob)
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
            #softmax_space_prob = activationBySoftmax(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            normalized_action_space_prob = normalize_list(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            #highest_prob_action_space_nr = np.random.choice(len(legal_next_states), p=softmax_space_prob)
            #highest_prob_action_space = legal_next_states[highest_prob_action_space_nr]
            #highest_prob = softmax_space_prob[highest_prob_action_space_nr]
            highest_prob = max(normalized_action_space_prob)
            index_highest_prob = np.where(normalized_action_space_prob == highest_prob)[0]
            if len(index_highest_prob) > 1:
                highest_prob_action_space = np.random.choice(legal_next_states, p=normalized_action_space_prob)
            else:
                highest_prob_action_space = legal_next_states[index_highest_prob[0] if len(index_highest_prob) > 1 else 0]
            return highest_prob_action_space, highest_prob

    def get_action(self, state):
        action_prob = self.policy_action.loc[state.hashIdentifier].to_numpy()
        highest_prob_action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return highest_prob_action, action_prob[highest_prob_action]

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
            discount = 0.95
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= gamma
            G_t[t] = G_sum
        return G_t

    def compute_J_avR(self, G_t):
        """
        calculate the objective function
        average reward per time-step
        purpose to measure the quality of a policy pi
        """
        return np.mean(G_t)
