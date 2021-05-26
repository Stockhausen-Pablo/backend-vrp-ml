import itertools
import collections
import numpy as np

from collections import namedtuple

import pandas as pd

from src.Utils.helper import normalize_list


class VRPAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent.
    Optimizes the policy function approximator using policy gradient.
    """

    EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_tours"])

    def __init__(self, env, policyManager, num_episodes, epsilon=0.5, alpha=0.5, gamma=0.9, epsilon_decay=1):
        self.env = env
        self.policyManager = policyManager
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.episode_statistics = VRPAgent.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_tours=[[] for i in range(num_episodes)]
        )
        self.episode = []
        self.Transition = collections.namedtuple("Transition",
                                                 ["state", "action", "action_prob", "reward", "next_state", "done"])

    def compute_G_t(self, episode, gamma):
        """
        args
          a list of rewards
        returns
          a list of cummulated rewards G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + .. + gamma^{T-t-1}*R_{T}
        """
        G_t = [0] * len(episode)

        for i, step_t in enumerate(episode):
            G_t[i] = gamma ** i * step_t.reward

        return G_t

    def train_model(self, gamma, epsilon, discountFactor):
        # Initialize a random policy
        state_hashes = self.env.getStateHashes()
        policy_prev = pd.DataFrame(index=state_hashes, columns=[self.env.actions])
        policy_prev.fillna(value=0.0, inplace=True)
        policy_prev = policy_prev + (1/len(self.env.actions))

        # Initialize Metainformation
        episodicRewards = []
        discountedRewards = []
        episodeHistory = []
        q_values = self.policyManager.q_values.copy()

        for epoch in range(self.num_episodes):
            self.runEpisode2(policy_prev, epoch, discountFactor)
            G_t = self.compute_G_t(self.episode, self.gamma)
            for i, step_t in enumerate(self.episode):
                c = step_t.state
                a = step_t.action
                n = step_t.next_state
                r = step_t.reward
                q_values[(c, a)] = self.policyManager.get_qvalue(c, a) + max(1 / self.n_s_a.get((c, a)), self.alpha) * (
                            G_t[i] - self.get_qvalue(c, a))


            episodicRewards.append(self.episode_statistics.episode_rewards[epoch])
            episodeHistory.append(self.episode)
            total_discounted_reward = sum(discountFactor ** i * step_t.reward for i, step_t in enumerate(self.episode))
            discountedRewards.append(total_discounted_reward)
            self.env.reset()
            policy_next, v = self.policyManager.policy_iteration(policy_prev, self.env, self.episode)
            if policy_next.equals(policy_prev):
                break
            policy_prev.update(policy_next)
        return self.episode_statistics

    def update(self, state, action, action_prob, reward, next_state, done):
        return self.episode.append(self.Transition(
            state=state, action=action, action_prob=action_prob, reward=reward, next_state=next_state, done=done))

    def getLegalActions(self):
        return self.env.getLegalActions()

    def observeTransition(self, action, actionNr):
        return self.env.step(action, actionNr)

    def observeTransition2(self, action):
        return self.env.takeStep(action)

    def runEpisode2(self, policy, epoch, discountFactor):
        state = self.env.reset()
        self.episode = []
        for step_t in itertools.count():
            action_probs = policy.loc[state.hashIdentifier].to_numpy()
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, currentTour, currentTours = self.observeTransition2(action)

            self.update(state, action, action_probs[action], reward, next_state, done)

            # Update statistics
            self.episode_statistics.episode_rewards[epoch] += reward
            self.episode_statistics.episode_lengths[epoch] = step_t
            self.episode_statistics.episode_tours[epoch] = currentTours

            # Print out which step is active
            print("\rStep {} @ Episode {}/{} ({})".format(
                step_t, epoch + 1, self.num_episodes, self.episode_statistics.episode_rewards[epoch]), end="")

            if done:
                break

            state = next_state


    def runEpisode(self, policy, epoch, discountFactor):
        state = self.env.reset()
        self.episode = []
        microhubCounter = 0
        for step_t in itertools.count():
            policy_best_action = 0.0
            legalActions, env_Action = self.getLegalActions()
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
            self.episode_statistics.episode_rewards[epoch] += reward
            self.episode_statistics.episode_lengths[epoch] = step_t
            self.episode_statistics.episode_tours[epoch] = currentTours

            # Print out which step is active
            print("\rStep {} @ Episode {}/{} ({})".format(
                step_t, epoch + 1, self.num_episodes, self.episode_statistics.episode_rewards[epoch]), end="")

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