import itertools

import numpy as np
import pandas as pd


class PolicyManager:
    def __init__(self, discountFactor=1.0, theta=0.00001):
        self.discountFactor = discountFactor
        self.theta = float(theta)

    def policy_eval(self, env, policy, values, gamma=0.9, theta=0.01):
        while True:
            delta = 0.0
            old_values = np.copy(values)
            for s in env.states:
                v = np.zeros_like(env.states, dtype=np.float32)
                for s_next in env.states:
                    #policy_action_prob = policy.action_prob fehlt hier TODO
                    p = env.getTransitionProbability(s.hashIdentifier, s_next.hashIdentifier)
                    r = env.reward_func(s, s_next)
                    v[s_next.stopid] = p * (r + gamma * values[s_next.stopid])

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

        for s in env.states:
            argmax_q = -np.inf
            best_a = None
            q_actual = 0
            for s_next in env.states:
                p = env.getTransitionProbability(s.hashIdentifier, s_next.hashIdentifier)
                r = env.reward_func(s, s_next)
                q_actual += p * (r + gamma * values[s_next.stopid])

                if q_actual > argmax_q:
                    argmax_q = q_actual
                    best_a = s_next.hashIdentifier

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

    def policy_iteration(self, env, episode, gamma=0.9, theta=0.01):
        # Initialize a random policy
        state_hashes = env.getStateHashes()
        policy = pd.DataFrame(index=state_hashes, columns=["best_a"])
        policy.fillna(value=0.0, inplace=True)
        print('Initial policy')
        # Initialize values to zero
        values = np.zeros_like(env.states, dtype=np.float32)
        # Run policy iteration
        policy_stable = False
        for i in itertools.count():
            print(f'Iteration {i}')
            V = self.policy_eval(env, policy, values)
            policy_stable = self.policy_improvement(V, env, policy, episode, gamma, self.discountFactor)
            if policy_stable:
                break

        return policy, values