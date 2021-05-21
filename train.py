import collections
import itertools

import numpy as np

from src.Utils import plotting
from src.Utils.helper import normalize_list

def train_model(
                env,
                num_episodes,
                discountFactor
                ):
    """REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient."""

    episodesStatistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_tours= [[] for i in range(num_episodes)]
    )

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for epoch in range(num_episodes):
        print(epoch)
        state = env.reset()
        episode = []
        for step_t in itertools.count():

            # Take a step
            legalActions = env.getLegalActions()
            action_probs = env.predict(state, legalActions)
            normalized_action_probs = normalize_list(action_probs)
            norm = sum(normalized_action_probs)

            actionBackToDepot = env.getMicrohubId()

            actionNr = np.random.choice(legalActions, p=normalized_action_probs)
            action = 1
            if actionNr == actionBackToDepot:
                action = 0
            next_state, reward, done, currentTour, currentTours = env.step(action, actionNr)

            # Keep track of the transition
            episode.append(Transition(
               state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            episodesStatistics.episode_rewards[epoch] += reward
            episodesStatistics.episode_lengths[epoch] = step_t
            episodesStatistics.episode_tours[epoch] = currentTours

            # Print out which step is active
            print("\rStep {} @ Episode {}/{} ({})".format(
                step_t, epoch + 1, num_episodes, episodesStatistics.episode_rewards[epoch]), end="")

            if done:
                break

            state = next_state

        for step_t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discountFactor**i * step_t.reward for i, step_t in enumerate(episode[step_t:]))
            # Calculate baseline/advantage
            #baseline_value = estimator_value.predict(transition.state)
            #advantage = total_return - baseline_value
            # Update our value estimator
            #estimator_value.update(transition.state, total_return)
            # Update our policy estimator
            #estimator_policy.update(transition.state, advantage, transition.action)

    return episodesStatistics