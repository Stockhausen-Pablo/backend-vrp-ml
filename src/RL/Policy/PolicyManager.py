import itertools

import numpy as np
import pandas as pd


class PolicyManager:
    def __init__(self, discountFactor=1.0, theta=0.00001):
        self.discountFactor = discountFactor
        self.theta = float(theta)

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

    # def policy_eval(self, values, env, policy, numIterations):
    #     # V = value function
    #     # V = np.zeros(len(self.env.states))
    #     V = {state.hashIdentifier: 0.0 for state in env.states}
    #     deltas = []
    #     for it in range(numIterations):
    #         delta = 0
    #         v = 0
    #         for policy_state in policy:
    #             action = policy_state.next_state.hashIdentifier
    #             action_prob = policy_state.action_prob
    #             prob = env.getTransitionProbability(policy_state.state.hashIdentifier, action)
    #             v += action_prob * prob * (policy_state.reward + self.discountFactor * V[action])  # Bellman-Equation
    #             delta = np.maximum(delta, np.abs(v - V[policy_state.state.hashIdentifier]))
    #             V[policy_state.state.hashIdentifier] = v
    #             deltas.append(delta)
    #         # if delta < self.theta:
    #         #    break
    #     return V

    def policy_improvement(self, values, env, policy, episode, gamma, discount_factor=1.0):
        def one_step_lookahead(state, V):
            state_hashes = env.getStateHashes()
            A = {state: 0.0 for state in state_hashes}
            for action in state_hashes:
                prob = env.getTransitionProbability(state, action)
                reward = env.reward_func_hash(state, action)
                A[action] += prob * (reward + discount_factor * V[action])
            return A

        policy_stable = True
        microhubCounter = 0
        microhubId = env.getMicrohubId()
        for s in env.states:
            argmax_q = -np.inf
            best_a = None
            q_actual = 0
            possibleStates = []
            if (s.hashIdentifier == microhubId):
                possibleStates = policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "possible_actions"]
            else:
                possibleStates = policy.at[s.hashIdentifier, "possible_actions"]
            for s_n in possibleStates:
                s_next = env.getStateByHash(s_n)
                p = env.getTransitionProbability(s.hashIdentifier, s_next.hashIdentifier)
                r = env.reward_func(s, s_next)
                q_actual += p * (r + gamma * values[s_next.stopid])

                if q_actual > argmax_q:
                    argmax_q = q_actual
                    best_a = s_next.hashIdentifier

            if s.hashIdentifier == microhubId:
                if policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "best_a"] != best_a:
                    policy_stable = False
                policy.at['{}/{}'.format(s.hashIdentifier, microhubCounter), "best_a"] = best_a
                microhubCounter += 1
            else:
                if policy.at[s.hashIdentifier, "best_a"] != best_a:
                    policy_stable = False
                policy.at[s.hashIdentifier, "best_a"] = best_a

        return policy_stable

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

    def policy_iteration(self, policy, env, episode, gamma=0.9, theta=0.01):
        # Initialize values to zero
        values = np.zeros_like(env.states, dtype=np.float32)
        # Run policy iteration
        policy_next = policy.copy()
        policy_stable = False
        for i in itertools.count():
            print(f'Iteration {i}')
            V = self.policy_eval(env, policy_next, values)
            policy_stable = self.policy_improvement(V, env, policy_next, episode, gamma, self.discountFactor)
            if policy_stable:
                break

        return policy_next, values
