import math
import random

import numpy as np
import pandas as pd

from src.Utils.helper import normalize_list, activationBySoftmax
import keras.backend as K

class PolicyManager:
    def __init__(self, state_hashes, actions, aco_boost_probability_Matrix, discountFactor=1.0, theta=0.00001):
        self.state_hashes = state_hashes
        self.discountFactor = discountFactor
        self.learning_rate = 0.09
        self.theta = float(theta)
        self.G = 0  # advantanges
        self.baseline_estimate = np.zeros_like(state_hashes, dtype=np.float32)
        # Initialize a random start policy
        # shows the distribution of a state over the action space per row
        #self.policy_action_space = aco_boost_probability_Matrix
        self.policy_action_space = pd.DataFrame(index=state_hashes, columns=state_hashes)
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

    def policy_update_by_learning(self, env, episode, gamma):
        # Run policy iteration
        policy_action_next = self.policy_action.copy()
        policy_action_space = self.policy_action_space.copy()
        policy_action_stable = False
        policy_action_space_stable = False
        # --------------------
        state_memory = np.array([step_t.state for step_t in episode])
        action_memory = np.array([step_t.action for step_t in episode])
        action_space_memory = np.array([step_t.next_state for step_t in episode])
        reward_memory = np.array([step_t.reward for step_t in episode])
        # --------------------
        G_t = self.compute_G_t(reward_memory, gamma)
        loseHistory =  []
        J_avR = self.compute_J_avR(G_t)
        std_deviation = np.std(G_t) if np.std(G_t) > 0 else 1
        self.G = (G_t - J_avR) / std_deviation  # G = list of advantages
        for idx, g in enumerate(self.G):
            # Find Current Weight for action
            current_weight = self.policy_action_space.at[episode[idx].state.hashIdentifier, episode[idx].next_state.hashIdentifier]
            current_weight = self.clip_weight(current_weight, 0.0001)  # to avoid zero division
            # Get possible lowest reward | goal to minimize reward (as lowest distance)
            earned_reward = episode[idx].reward
            test = self.calculate_value_func(env, episode[idx], self.policy_action_space.loc[episode[idx].state.hashIdentifier, :], gamma)
            baseline = episode[idx].possible_rewards.min()
            advantage = earned_reward - baseline
            # Calculate Lose
            lose = self.custom_loss(g, current_weight, advantage)
            loseHistory.append(lose)
            # start back-propagation
            # Backpropagation mit Trägheitsterm fehlt
            grad_weight = -self.learning_rate * (lose / current_weight)
            weight_new = self.resolve_weight(grad_weight, current_weight, episode[idx].possible_rewards)

            if (1.0 > weight_new > 0.0):
                self.policy_action_space.at[
                    episode[idx].state.hashIdentifier, episode[idx].next_state.hashIdentifier] = weight_new
            else:
                self.policy_action_space.at[
                    episode[idx].state.hashIdentifier, episode[idx].next_state.hashIdentifier] = current_weight - (current_weight / 2)

        # lrate = initial_lrate * (1 / (1 + decay * iteration))
        return G_t, J_avR, loseHistory
    
    def calculate_value_func(self, env, episode, weights, gamma, theta=0.01):
        while True:
            delta = 0.0
            old_values = np.copy(self.baseline_estimate)
            for idx_v, s_a in enumerate(self.state_hashes):
                v = np.zeros_like(episode.possible_next_states, dtype=np.float32)
                for idx_n, s_n in enumerate(episode.possible_next_states):
                    s_next = env.getStateByHash(s_n)
                    pi_s_a = 1
                    p = weights.loc[s_n:]
                    r = env.reward_func_hash(s_a, s_n)
                    v[s_next.stopid] = pi_s_a * p * (r + gamma * self.baseline_estimate[s_next.stopid])
                self.values[idx_v] = np.sum(v)
                delta = np.maximum(delta, np.absolute(self.values[idx_v] - old_values[idx_v]))
            if delta < theta:
                break
        return self.values

    def resolve_weight(self, grad_weight, current_weight, possible_rewards):
        weight_new = 0.0
        if grad_weight == 0 and len(possible_rewards) > 1:
            weight_new = current_weight ** 0.9  # try to increase probability by 10%
        if grad_weight == 0 and len(possible_rewards) == 1:
            weight_new = current_weight
        if grad_weight > 0 and len(possible_rewards) == 1:
            weight_new = current_weight
        if grad_weight > 0 and len(possible_rewards) > 1:
            weight_new = current_weight + grad_weight
        return weight_new

    def clip_weight(self, current_weight, clipValue):
        if current_weight == 1:
            current_weight = current_weight - clipValue
        if current_weight == 0:
            current_weight = current_weight + clipValue
        return current_weight
    
    def custom_loss(self, y_true, y_pred, advantages):
        log_lik = y_true * K.log(y_pred)
        return K.mean(-log_lik * advantages)

    def get_action_space(self, eps, state, legal_next_states):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            highest_prob_action_space = random.choice(legal_next_states)
            return highest_prob_action_space, 1
        else:
            action_space_prob = self.policy_action_space.loc[state.hashIdentifier, legal_next_states].to_numpy()
            softmax_space_prob = activationBySoftmax(action_space_prob) if len(legal_next_states) > 1 else [1.0]
            highest_prob_action_space_nr = np.random.choice(len(legal_next_states), p=softmax_space_prob)
            highest_prob_action_space = legal_next_states[highest_prob_action_space_nr]
            highest_prob = softmax_space_prob[highest_prob_action_space_nr]
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
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
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
