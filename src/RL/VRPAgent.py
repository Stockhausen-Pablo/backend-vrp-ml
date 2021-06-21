import collections
from collections import namedtuple

import numpy as np


class VRPAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent.
    Optimizes the policy function approximator using policy gradient.
    Agent is set to On-Policy behaviour / ( to delete -> Since one full trajectory must be completed to construct a sample space)
    """

    EpisodeStats = namedtuple("Stats",
                              ["episode_lengths", "episode_rewards", "episode_tours", "episode_G_t", "episode_J_avR",
                               "episode_policy_reward"])

    def __init__(self, env, policyManager, num_episodes, max_steps, discount_factor, eps=0.15):
        # Given by main
        self.env = env
        self.policyManager = policyManager
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = discount_factor

        self.eps = eps

        self.episode = []

        self.Transition = collections.namedtuple("Transition",
                                                 ["state",
                                                  "action",
                                                  "action_space_prob",
                                                  "reward",
                                                  "next_state",
                                                  "done",
                                                  "possible_next_states",
                                                  "possible_next_states_hub_ignored",
                                                  "possible_rewards",
                                                  "microhub_counter"
                                                  ]
                                                 )
        self.episode_statistics = VRPAgent.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_tours=[[] for i in range(num_episodes)],
            episode_G_t=np.zeros(num_episodes),
            episode_J_avR=np.zeros(num_episodes),
            episode_policy_reward=np.zeros(num_episodes)
        )

    def train_model(self):
        for epoch in range(self.num_episodes):
            self.runEpisode(epoch)

            G_t, J_avR, loseHistory, eps, policy_reward = self.policyManager.policy_update_by_learning(self.env,
                                                                                                       self.episode,
                                                                                                       self.episode_statistics.episode_rewards[
                                                                                                           epoch],
                                                                                                       self.gamma,
                                                                                                       self.max_steps,
                                                                                                       self.num_episodes,
                                                                                                       epoch)

            # Update Metainformation
            self.episode_statistics.episode_G_t[epoch] = sum(G_t)
            self.episode_statistics.episode_J_avR[epoch] = J_avR
            self.episode_statistics.episode_policy_reward[epoch] = policy_reward

            self.eps = eps
            self.env.reset()

        best_policy_reward = min(self.episode_statistics.episode_policy_reward)
        worst_policy_reward = max(self.episode_statistics.episode_policy_reward)
        last_policy_reward = self.episode_statistics.episode_policy_reward[
            len(self.episode_statistics.episode_policy_reward) - 1]

        return self.episode_statistics, \
               self.policyManager.policy_action_space, \
               best_policy_reward, \
               worst_policy_reward, \
               last_policy_reward

    def update(self, state, action, action_space_prob, reward, next_state, done, possible_next_states,
               possible_next_states_hub_ignored,
               possible_rewards, microhub_counter):
        return self.episode.append(
            self.Transition(
                state=state,
                action=action,
                action_space_prob=action_space_prob,
                reward=reward,
                next_state=next_state,
                done=done,
                possible_next_states=possible_next_states,
                possible_next_states_hub_ignored=possible_next_states_hub_ignored,
                possible_rewards=possible_rewards,
                microhub_counter=microhub_counter
            )
        )

    def get_possible_rewards_at_t(self, state, action_space_list):
        return self.env.possible_rewards(state, action_space_list)

    def getLegalAction(self):
        return self.env.getLegalAction()

    def observeTransition(self, action, action_space):
        return self.env.step(action, action_space)

    def runEpisode(self, epoch):
        # --------------------
        # PREPARE EPOCH RUN
        state = self.env.reset()
        self.episode = []

        for step_t in range(self.max_steps):
            # --------------------
            # LEGAL NEXT STATES
            # given by the environment
            legal_next_action, legal_next_states, legal_next_states_hubs_ignored, microhub_counter = self.getLegalAction()
            possible_rewards = self.get_possible_rewards_at_t(state.hashIdentifier,
                                                              legal_next_states if legal_next_action == 1 else [
                                                                  self.env.get_microhub_hash()])

            # --------------------
            # CHOOSE ACTION SPACE
            action_space, action_space_prob = self.policyManager.get_action_space(self.eps, state, legal_next_states,
                                                                                  microhub_counter)

            # --------------------
            # DO STEP IN ENVIRONMENT
            # follow policy
            next_state, reward, done, currentTour, currentTours = self.observeTransition(legal_next_action,
                                                                                         action_space)

            # --------------------
            # SAVE TRANSITION
            self.update(state, legal_next_action, action_space_prob, reward, next_state, done, legal_next_states,
                        legal_next_states_hubs_ignored,
                        possible_rewards, microhub_counter)

            # --------------------
            # UPDATE EPISODE STATISTICS
            self.episode_statistics.episode_rewards[epoch] += reward
            self.episode_statistics.episode_lengths[epoch] = step_t
            self.episode_statistics.episode_tours[epoch] = currentTours

            # --------------------
            # PRINT EPOCH PROGRESS
            print("Step {} @ Episode {}/{} ({})\n".format(
                step_t, epoch + 1, self.num_episodes, self.episode_statistics.episode_rewards[epoch]), end="")

            if done:
                break

            state = next_state
