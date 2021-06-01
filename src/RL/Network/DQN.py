import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        # Creating layers for the learnable theta parameters in our action-value funtion
        self.theta1 = nn.Linear(5, 5, True)
        self.theta2 = nn.Linear(5, 5, True)
        self.theta3 = nn.Linear(5, 5, True)
        self.theta4 = nn.Linear(1, 5, True)
        self.theta5 = nn.Linear(2 * 5, 1, True)
        self.theta6 = nn.Linear(5, 5, True)
        self.theta7 = nn.Linear(5, 5, True)

    # Forward pass through the network
    def forward(self, features, nodeLocations):
        # Calculate the first term of our action-value function which is based on the features of the node
        term1 = self.theta1(features)
        for layer in self.padding:
            term1 = layer(F.relu(term1))
        # The Third term of our equation which is an aggregate of distances of all the neighboring nodes
        term3 = self.theta3(torch.sum(F.relu(self.theta4(nodeLocations.unsqueeze(3))), dim=1))

        # The second term which is the aggregate of embeddings of all neighboring nodes iterated for T times
        connections = torch.where(nodeLocations > 0, torch.ones_like(nodeLocations),
                                  torch.zeros_like(nodeLocations)).to(device)
        mu = torch.zeros(features.shape[0], (config.numCust + config.numDepot), config.embeddingDimension,
                         device=device)
        for t in range(config.embeddingIteration):
            term2 = self.theta2(connections.matmul(mu))
            mu = F.relu(term1 + term2 + term3)

        # Aggregate of final embeddings of neighbors
        term6 = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, (config.numCust + config.numDepot), 1))
        term7 = self.theta7(mu)
        # Final action value function result
        concat = F.relu(torch.cat([term6, term7], dim=2))
        return self.theta5(concat).squeeze(dim=2)