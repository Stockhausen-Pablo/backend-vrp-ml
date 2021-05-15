

class VRPAgent:
    def __init__(self):
        self.num_iterations = 250
        self.collect_episodes_per_iteration = 2
        replay_buffer_capacity = 2000