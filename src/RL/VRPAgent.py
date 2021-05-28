import collections
from collections import namedtuple

import numpy as np


class VRPAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent.
    Optimizes the policy function approximator using policy gradient.
    Agent is set to Off-Policy behaviour / Since one full trajectory must be completed to construct a sample space
    """

    EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_tours", "episode_G_t", "episode_J_avR"])

    def __init__(self, env, policyManager, num_episodes, alpha=0.5, gamma=0.95, eps=0.05):
        self.env = env
        self.policyManager = policyManager
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.episode = []

        self.max_steps = 10000

        self.Transition = collections.namedtuple("Transition",
                                                 ["state",
                                                  "action",
                                                  "action_space_prob",
                                                  "reward",
                                                  "next_state",
                                                  "done",
                                                  "possible_next_states",
                                                  "possible_rewards"
                                                  ]
                                                 )
        self.episode_statistics = VRPAgent.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_tours=[[] for i in range(num_episodes)],
            episode_G_t=np.zeros(num_episodes),
            episode_J_avR=np.zeros(num_episodes)
        )

    def train_model(self):
        for epoch in range(self.num_episodes):
            self.runEpisode(epoch)
            G_t, J_avR, loseHistory = self.policyManager.policy_update_by_learning(self.env, self.episode, self.gamma)

            # Update Metainformation
            self.episode_statistics.episode_G_t[epoch] = sum(G_t)
            self.episode_statistics.episode_J_avR[epoch] = J_avR

            self.env.reset()

        return self.episode_statistics, self.policyManager.policy_action_space

    def update(self, state, action, action_space_prob, reward, next_state, done, possible_next_states,
               possible_rewards):
        return self.episode.append(
            self.Transition(
                state=state,
                action=action,
                action_space_prob=action_space_prob,
                reward=reward,
                next_state=next_state,
                done=done,
                possible_next_states=possible_next_states,
                possible_rewards=possible_rewards
            )
        )

    def get_possible_rewards_at_t(self, state, action_space_list):
        return self.env.possible_rewards(state, action_space_list)

    def getLegalAction(self):
        return self.env.getLegalAction()

    def observeTransition(self, action, action_space):
        return self.env.step(action, action_space)

    def runEpisode(self, epoch):
        state = self.env.reset()
        self.episode = []
        for step_t in range(self.max_steps):
            # Check Policy Estimation
            legal_next_action, legal_next_states = self.getLegalAction()
            possible_rewards = self.get_possible_rewards_at_t(state.hashIdentifier, legal_next_states)
            action_space, action_space_prob = self.policyManager.get_action_space(self.eps, state, legal_next_states)

            # take a step by policy
            next_state, reward, done, currentTour, currentTours = self.observeTransition(legal_next_action,
                                                                                         action_space)

            self.update(state, legal_next_action, action_space_prob, reward, next_state, done, legal_next_states,
                        possible_rewards)

            # Update statistics
            self.episode_statistics.episode_rewards[epoch] += reward
            self.episode_statistics.episode_lengths[epoch] = step_t
            self.episode_statistics.episode_tours[epoch] = currentTours

            # Print out which step is active
            print("Step {} @ Episode {}/{} ({})\n".format(
                step_t, epoch + 1, self.num_episodes, self.episode_statistics.episode_rewards[epoch]), end="")

            if done:
                break

            state = next_state

        # for step_t, transition in enumerate(self.episode):
        # The return after this timestep
        # G_t
        # total_discounted_reward = sum(discountFactor ** i * step_t.reward for i, step_t in enumerate(self.episode[step_t:]))
        # Calculate baseline/advantage
        # V = policyManager.policy_eval(transition)
        # baseline_value = self.estimator_value.predict(transition.state)
        # advantage = total_discounted_reward - baseline_value
        # Update our value estimator
        # estimator_value.update(transition.state, total_return)
        # Update our policy estimator
        # estimator_policy.update(transition.state, advantage, transition.action)
