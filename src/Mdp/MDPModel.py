class MDPModel:
    def __init__(self,
                 actionList,
                 transitionMatrix=None,
                 reward=None,
                 states=None,
                 gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.states = states
        self.actionList = actionList
        self.transitionMatrix = transitionMatrix
        self.gamma = gamma
        self.reward = reward or {s: 0 for s in self.states}

    def getReward(self, current_stop, next_stop):
        reward = self.reward.at[current_stop.stopid, next_stop.stopid]
        return reward

    def getTransitionModel(self, state, action):
        """return a list of (probability, result-state) pairs"""
        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def getActions(self, state):
        return self.actionList

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None