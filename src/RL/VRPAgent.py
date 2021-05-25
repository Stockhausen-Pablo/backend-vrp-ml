import itertools
import collections
import numpy as np

from collections import namedtuple

import pandas as pd

from src.Utils.helper import normalize_list


class VRPAgent:
    EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_tours"])

    def __init__(self, env, policyManager, num_episodes):
        self.env = env
        self.policyManager = policyManager
        self.num_episodes = num_episodes
        self.collect_episodes_per_iteration = 2
        self.episodesStatistics = VRPAgent.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_tours=[[] for i in range(num_episodes)]
        )
        self.episode = []
        self.Transition = collections.namedtuple("Transition",
                                                 ["state", "action", "action_prob", "reward", "next_state", "done"])

    def train_model(self, gamma, epsilon, discountFactor):
        episodeList = []

        # Initialize a random policy
        state_hashes = self.env.getStateHashes()
        policy_prev = pd.DataFrame(index=state_hashes[1:], columns=["best_a", "possible_actions"])
        policy_prev.fillna(value=0.0, inplace=True)
        policy_prev["possible_actions"] = policy_prev["possible_actions"].astype('object')
        for epoch in range(self.num_episodes):
            policy_prev = self.startEpisode(policy_prev, epoch, discountFactor)
            self.env.reset()
            episodeList.append(self.episode)
            total_discounted_reward = sum(discountFactor ** i * step_t.reward for i, step_t in enumerate(self.episode))
            policy_next, v = self.policyManager.policy_iteration(policy_prev, self.env, self.episode)
            if policy_next.equals(policy_prev):
                break
            policy_prev.update(policy_next)
        return self.episodesStatistics

    def update(self, state, action, action_prob, reward, next_state, done):
        return self.episode.append(self.Transition(
            state=state, action=action, action_prob=action_prob, reward=reward, next_state=next_state, done=done))

    def getLegalActions(self):
        return self.env.getLegalActions()

    def observeTransition(self, action, actionNr):
        return self.env.step(action, actionNr)

    def startEpisode(self, policy, epoch, discountFactor):
        state = self.env.reset()
        self.episode = []
        microhubCounter = 0
        for step_t in itertools.count():
            policy_best_action = 0.0
            legalActions = self.getLegalActions()
            action_prob = 1/len(legalActions)
            actionBackToDepot = self.env.getMicrohubId()
            if state.hashIdentifier == actionBackToDepot:
                if '{}/{}'.format(state.hashIdentifier,microhubCounter) in policy.index.values:
                    policy_best_action = policy.at['{}/{}'.format(state.hashIdentifier,microhubCounter), "best_a"]
                    policy.at['{}/{}'.format(state.hashIdentifier,microhubCounter), "possible_actions"] = legalActions
                else:
                    new_row = pd.Series(data={'best_a': 0.0, 'possible_actions': legalActions}, name='{}/{}'.format(state.hashIdentifier,microhubCounter))
                    policy = policy.append(new_row, ignore_index=False)
            else:
                policy_best_action = policy.at[state.hashIdentifier, "best_a"]
                policy.at[state.hashIdentifier, "possible_actions"] = legalActions
            actionNr = 0.0
            if policy_best_action in legalActions:
                actionNr = policy_best_action
            else:
                action_probs = self.env.predict(state, legalActions)
                normalized_action_probs = normalize_list(action_probs)
                actionNr = np.random.choice(legalActions, p=normalized_action_probs)
                if state.hashIdentifier == actionBackToDepot:
                    policy.at['{}/{}'.format(state.hashIdentifier, microhubCounter), "best_a"] = actionNr
                else:
                    policy.at[state.hashIdentifier, "best_a"] = actionNr
            action = 1
            if actionNr == actionBackToDepot:
                action = 0
            next_state, reward, done, currentTour, currentTours = self.observeTransition(action, actionNr)

            self.update(state, action, action_prob, reward, next_state, done)

            # Update statistics
            self.episodesStatistics.episode_rewards[epoch] += reward
            self.episodesStatistics.episode_lengths[epoch] = step_t
            self.episodesStatistics.episode_tours[epoch] = currentTours

            # Print out which step is active
            print("\rStep {} @ Episode {}/{} ({})".format(
                step_t, epoch + 1, self.num_episodes, self.episodesStatistics.episode_rewards[epoch]), end="")

            if state.hashIdentifier == actionBackToDepot:
                microhubCounter += 1
            if done:
                break

            state = next_state


        # V = policyManager.policy_eval(self.episodes)

        # for episode in self.episodes:
        #    V = policyManager.policy_eval(episode)

        for step_t, transition in enumerate(self.episode):
            # The return after this timestep
            # G_t
            total_discounted_reward = sum(discountFactor ** i * step_t.reward for i, step_t in enumerate(self.episode[step_t:]))
            # Calculate baseline/advantage
            # V = policyManager.policy_eval(transition)
            # baseline_value = estimator_value.predict(transition.state)
            # advantage = total_return - baseline_value
            # Update our value estimator
            # estimator_value.update(transition.state, total_return)
            # Update our policy estimator
            # estimator_policy.update(transition.state, advantage, transition.action)
        return policy