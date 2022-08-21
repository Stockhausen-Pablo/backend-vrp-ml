import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


# For more details, see:
# https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#creating_your_own_python_environment
class VrpEnv(py_environment.PyEnvironment):
    def __init__(self, microhub, observation_size, initial_stops):
        # action space that the agent observes to choose next action
        # 0: choose next stop
        # 1: end tour with >1 stops being left
        # 2: end tour with 0 stops left
        # shape == 2: (action_id in [0], selected_stop in [1])
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.int32,
            minimum=0,
            maximum=2,
            name='action'
        )

        #  Observation from the selected action
        # needs to include all possible next stops
        # should evaluate current load
        # https://github.com/tensorflow/agents/blob/master/tf_agents/specs/array_spec_test.py
        self._observation_spec = {
            'observations': array_spec.ArraySpec(
                shape=(),
                dtype=dict,
                name='observation'
            ),
            'legal_next_moves': array_spec.ArraySpec(
                shape=(observation_size,),
                dtype=dict,
                name='legal_next_moves'
            )
        }

        # self._observation_spec = {'observations': array_spec.ArraySpec(shape=(100,), dtype=np.float64),
        #                          'legal_moves': array_spec.ArraySpec(shape=(self.num_moves(),), dtype=np.bool_),
        #                          'other_output1': array_spec.ArraySpec(shape=(10,), dtype=np.int64),
        #                          'other_output2': array_spec.ArraySpec(shape=(2, 5, 2), dtype=np.int64)}

        self._state = microhub
        self._episode_ended = False

        self._microhub = microhub
        self._initial_stops = initial_stops  # stops_ids = [item['id'] for item in stops]

        # tour data
        self._tours = []

        self._current_tour = []
        self._current_weight = 0.0
        self._current_volume = 0.0
        self._current_distance = 0.0

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def get_info(self):
        pass

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def _step(self, action):
        # Check if episode ended
        if self._episode_ended:
            return self.reset()

        if action[0] == 2:
            self._episode_ended = True
        elif action[0] == 1:
            # vehicle capacity reached
            # drive back to hub
            # start new tour
            pass
        elif action[0] == 0:
            # retrieve selected stop
            # evaluate if selected stop is allowed
            # update current state
            # remove selected state from _possible stops
            pass
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended:
            # calculate overall rewards
            reward = 10
            return ts.termination({'observations': np.array(self._state, dtype=dict), 'legal_next_moves': np.array(self._initial_stops, dtype=dict)}, reward)
        else:
            return ts.transition(observation = {'observations': np.array(self._state, dtype=dict), 'legal_next_moves': np.array(self._initial_stops, dtype=dict)}, reward=0.0, discount=1.0)

    def _reset(self):
        self._state = self._microhub
        self._current_tour.append(self._microhub)
        self._episode_ended = False
        return ts.restart(observation = {'observations': np.array(self._state, dtype=dict), 'legal_next_moves': np.array(self._initial_stops, dtype=dict)})

    def transition(self, next_state):
        self._state = next_state
        self._current_tour.append(next_state)
        self.update_meta(next_state)
        self.update_observation_spec(next_state)

    def update_meta(self, next_state):
        self._current_weight += next_state.get('demandWeight')
        self._current_volume += next_state.get('demandVolume')

    def update_observation_spec(self, next_state):
        self._initial_stops = [item for item in self._initial_stops if item['id'] != next_state.get('id')]
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self._initial_stops),),
            dtype=dict,
            name='observation'
        )
