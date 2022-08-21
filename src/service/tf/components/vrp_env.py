import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


# For more details, see:
# https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#creating_your_own_python_environment
class VrpEnv(py_environment.PyEnvironment):
    def __init__(self, microhub, observation_size, initial_stops):
        # action space that the agent observes to choose next action
        # 0: end tour
        # 1: choose next stop
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=1,
            name='action'
        )

        #  Observation from the selected action
        # needs to include all possible next stops
        # should evaluate current load
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observation_size,),
            dtype=np.int32,
            name='observation'
        )

        self._state = microhub
        self._episode_ended = False

        self._microhub = microhub
        self._initial_stops = initial_stops  # stops_ids = [item['id'] for item in stops]

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

        if action == 1 & (not self._possible_stops):
            self._episode_ended = True
        elif action == 0:
            # retrieve selected stop
            # evaluate if selected stop is allowed
            # update current state
            # remove selected state from _possible stops
            pass
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or (not self._possible_stops):
            # calculate overall rewards
            #
            pass
        else:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    def _reset(self):
        self._state = self._microhub
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
