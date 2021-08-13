import collections
from collections import namedtuple

import requests
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

    def __init__(self,
                 env,
                 policy_manager,
                 num_episodes,
                 max_steps,
                 discount_factor,
                 eps=0.15):

        # --------------------
        # GIVEN INSTANCES
        self.env = env
        self.policy_manager = policy_manager

        # --------------------
        # SETTINGS
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = discount_factor
        self.eps = eps

        # --------------------
        # EPISODE META
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
            episode_rewards=np.empty(num_episodes) * np.nan,
            episode_tours=[[] for i in range(num_episodes)],
            episode_G_t=np.zeros(num_episodes),
            episode_J_avR=np.zeros(num_episodes),
            episode_policy_reward=np.empty(num_episodes) * np.nan
        )

    def update_training_stream(self, epoch, policy_reward, sum_G_t, best_policy_reward, worst_policy_reward, policy_tours):
        pload = {
            'epoch': epoch,
            'policy_reward': policy_reward,
            'sum_G_t': sum_G_t,
            'best_policy_reward': best_policy_reward,
            'worst_policy_reward': worst_policy_reward,
            'policy_tours': policy_tours
        }
        headers = {'Content-type': 'form-data'}
        r = requests.post('http://127.0.0.1:5000/ml-service/training/update', data=pload)
        print(r.text)

    def train_model(self) -> object:
        """
        START TRAINING THE ML-MODEL
        :return: episode statistics, policy action space, best policy reward, worst policy reward, last policy reward
        """
        for epoch in range(self.num_episodes):
            self.run_episode(epoch)
            G_t, J_avR, loseHistory, eps, policy_reward, policy_tours = self.policy_manager.policy_update_by_learning(self.env,
                                                                                                        self.episode,
                                                                                                        self.episode_statistics.episode_rewards[
                                                                                                            epoch],
                                                                                                        self.gamma,
                                                                                                        self.max_steps,
                                                                                                        self.num_episodes,
                                                                                                        epoch)

            # Update Meta information
            self.episode_statistics.episode_G_t[epoch] = sum(G_t)
            self.episode_statistics.episode_J_avR[epoch] = J_avR
            self.episode_statistics.episode_policy_reward[epoch] = policy_reward

            self.eps = eps
            self.update_training_stream(
                epoch,
                policy_reward,
                sum(G_t),
                min(self.episode_statistics.episode_policy_reward),
                max(self.episode_statistics.episode_policy_reward),
                policy_tours
            )
            self.env.reset()

        best_policy_reward = min(self.episode_statistics.episode_policy_reward)
        worst_policy_reward = max(self.episode_statistics.episode_policy_reward)
        last_policy_reward = self.episode_statistics.episode_policy_reward[
            len(self.episode_statistics.episode_policy_reward) - 1]

        return self.episode_statistics, \
               self.policy_manager.policy_action_space, \
               best_policy_reward, \
               worst_policy_reward, \
               last_policy_reward

    def update(self, state: object, action: object, action_space_prob: object, reward: float, next_state: object,
               done: bool, possible_next_states: object,
               possible_next_states_hub_ignored: object,
               possible_rewards: object, microhub_counter: int) -> object:
        """
        :param state: current state
        :param action: taken action
        :param action_space_prob: taken action space prob
        :param reward: received reward
        :param next_state: following state
        :param done: done boolean
        :param possible_next_states: list of possible next states
        :param possible_next_states_hub_ignored: lost of possible next states without microhub counter
        :param possible_rewards: list of possible alternative rewards
        :param microhub_counter: counter number specifying how often the tour constructor returned to the hub
        :return: Transition
        """
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

    def get_possible_rewards_at_t(self, state: object, action_space_list: object) -> object:
        """
        :param state: current state hash_id
        :param action_space_list: list of possible next states hash_ids
        :return: list of possible rewards
        """
        return self.env.possible_rewards(state, action_space_list)

    def get_legal_action(self) -> object:
        """
        :return: action, legal next states, legal next states with hub counter ignored, local search distances,
        bin packing capacities, microhub counter
        """
        return self.env.get_next_legal_action()

    def observe_transition(self, action, action_space):
        """
        :param action: taken action
        :param action_space: given action space
        :return: next state, reward, done, current tour, all tours
        """
        return self.env.step(action, action_space)

    def run_episode(self, epoch: int) -> object:
        """
        :param epoch: current epoch count
        :return: None
        """
        # --------------------
        # PREPARE EPOCH RUN
        state = self.env.reset()
        self.episode = []

        for step_t in range(self.max_steps):
            # --------------------
            # LEGAL NEXT STATES
            # given by the environment
            legal_next_action, legal_next_states, legal_next_states_hubs_ignored, legal_next_states_local_search_distance, legal_next_states_bin_packing_capacities, microhub_counter = self.get_legal_action()
            possible_rewards = self.get_possible_rewards_at_t(state.hash_id,
                                                              legal_next_states if legal_next_action == 1 else [
                                                                  self.env.get_microhub_hash()])

            # --------------------
            # CHOOSE ACTION SPACE
            action_space, action_space_prob = self.policy_manager.get_action_space(self.eps,
                                                                                   state,
                                                                                   legal_next_states,
                                                                                   legal_next_states_local_search_distance,
                                                                                   legal_next_states_bin_packing_capacities,
                                                                                   microhub_counter)

            # --------------------
            # DO STEP IN ENVIRONMENT
            # follow policy
            next_state, reward, done, current_tour, current_tours = self.observe_transition(legal_next_action,
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
            self.episode_statistics.episode_tours[epoch] = current_tours

            # --------------------
            # PRINT EPOCH PROGRESS
            print("Step {} @ Episode {}/{} ({})\n".format(
                step_t, epoch + 1, self.num_episodes, self.episode_statistics.episode_rewards[epoch]), end="")

            if done:
                break

            state = next_state
