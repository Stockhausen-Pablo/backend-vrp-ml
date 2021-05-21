import itertools
import collections
import numpy as np

from collections import namedtuple

from src.Utils.helper import normalize_list

class VRPAgent:

    EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_tours"])

    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.collect_episodes_per_iteration = 2
        self.episodesStatistics = VRPAgent.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_tours=[[] for i in range(num_episodes)]
        )
        self.episode = []
        self.Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    def train_model(self, gamma, epsilon, discountFactor):
        for epoch in range(self.num_episodes):
            self.startEpisode(epoch, discountFactor)
        return self.stopEpisode()

    def update(self, state, action, reward, next_state, done):
        return self.episode.append(self.Transition(
            state=state, action=action, reward=reward, next_state=next_state, done=done))

    def getLegalActions(self):
        return self.env.getLegalActions()

    def observeTransition(self, action, actionNr):
        return self.env.step(action, actionNr)

    def startEpisode(self, epoch, discountFactor):
        state = self.env.reset()
        self.episode = []
        for step_t in itertools.count():

            legalActions = self.getLegalActions()
            action_probs = self.env.predict(state, legalActions)
            normalized_action_probs = normalize_list(action_probs)
            norm = sum(normalized_action_probs)

            actionBackToDepot = self.env.getMicrohubId()

            actionNr = np.random.choice(legalActions, p=normalized_action_probs)
            action = 1
            if actionNr == actionBackToDepot:
                action = 0
            next_state, reward, done, currentTour, currentTours = self.observeTransition(action, actionNr)

            self.update(state, action, reward, next_state, done)

            # Update statistics
            self.episodesStatistics.episode_rewards[epoch] += reward
            self.episodesStatistics.episode_lengths[epoch] = step_t
            self.episodesStatistics.episode_tours[epoch] = currentTours

            # Print out which step is active
            print("\rStep {} @ Episode {}/{} ({})".format(
                step_t, epoch + 1, self.num_episodes, self.episodesStatistics.episode_rewards[epoch]), end="")

            if done:
                break

            state = next_state

        for step_t, transition in enumerate(self.episode):
            # The return after this timestep
            total_return = sum(discountFactor**i * step_t.reward for i, step_t in enumerate(self.episode[step_t:]))
            # Calculate baseline/advantage
            #baseline_value = estimator_value.predict(transition.state)
            #advantage = total_return - baseline_value
            # Update our value estimator
            #estimator_value.update(transition.state, total_return)
            # Update our policy estimator
            #estimator_policy.update(transition.state, advantage, transition.action)

    def stopEpisode(self):
        return self.episodesStatistics