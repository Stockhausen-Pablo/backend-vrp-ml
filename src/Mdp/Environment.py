

class Environment:

    def __init__(self, states, actions, probabilityMatrix, rewardFunction, microHub, discountFactor, capacityDemand,
                 vehicles, vehicleCapacity):
        self.states = states
        self.actions = actions
        self.probabilityMatrix = probabilityMatrix
        self.rewardFunction = rewardFunction
        self.microHub = microHub
        self.discountFactor = discountFactor
        self.capacityDemand = capacityDemand
        self.vehicles = vehicles
        self.vehicleCapacities = vehicleCapacity

        #def step(self, action):
        #def reset(self):
        #def render(self, mode='human'):
        #def close(self):