import csv
from argsConfig import getParams

import src.Tour.TourManager as tManager

from src.Tour.Stop import Stop
from src.Utils.plotter import plotCoordinates
from src.Utils.helper import normalize_df
from src.Utils import plotting
from src.Mdp.VRPEnvironment import VRPEnvironment
from src.RL.VRPAgent import VRPAgent
from src.Aco.AntManager import AntManager
from src.RL.Policy.PolicyManager import PolicyManager



def loadStopData(dataSet):
    with open('data/stops/' + dataSet + '.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            tManager.addStop(
                Stop(float(row[0]), int(row[1]), float(row[3]), float(row[2]), int(row[4]), int(row[5])))
    tManager.calculateDistances()
    tManager.calculateDistanceMatrix()
    tManager.initCapacityDemands()


def main(args):
    # Input
    print("---------System menu---------")
    print("Below please specify the configuration options of the program")
    dataSet = input("Please specify the data source of the stops to be processed:")
    amountVehicles = int(input("How many vehicles will be used:"))
    capacityWeight = float(input("What is the maximum weight that the vehicle can carry:"))
    capacityVolume = float(input("What is the maximum volume that the vehicle can hold:"))

    # -----
    tManager.clear()

    # Load Stop Data
    loadStopData(dataSet)

    # Setup Distance Matrix for later use
    distanceMatrix = tManager.getDistances()

    # Plot Coordinates of input stops
    plotCoordinates()

    if args['train']:
        print("-Entered Training Mode-")

        # ------------------
        # Setting up and Running ACO
        print("-Starting up Ant Colony Optimization to get Probability Matrix-")
        antManager = AntManager(
            stops=tManager.getListOfStops(),
            start_stop=tManager.getStop(0),
            vehicleWeight=capacityWeight,
            vehicleVolume=capacityVolume,
            vehicleCount=amountVehicles,
            discountAlpha=.5,
            discountBeta=1.2,
            pheromone_evaporation_coefficient=.40,
            pheromone_constant=1.0,
            iterations=80
        )
        # ------------------
        # Retrieving solution from ACO and preparing further transformation

        resultACO = antManager.runACO()
        ant_shortest_distance = resultACO[0]
        ant_shortest_path = resultACO[1]
        ant_df_pheromoneMatrix = resultACO[2]
        ant_probability_Matrix = resultACO[3]

        # Normalize
        normalized_probability_Matrix = normalize_df(ant_probability_Matrix)

        # ------------------
        # Setting up MDP-Environment
        environment = VRPEnvironment(
            states=tManager.getListOfStops(),
            actions=[[0, 1], [1], [0]],
            probabilityMatrix=normalized_probability_Matrix,
            distanceMatrix=distanceMatrix,
            rewardFunction=0,
            microHub=tManager.getMicrohub(),
            capacityDemands=tManager.getCapacityDemands(),
            vehicles=amountVehicles,
            vehicleWeight=capacityWeight,
            vehicleVolume=capacityVolume
        )

        policyManager = PolicyManager()

        agent = VRPAgent(env= environment,
                         policyManager=policyManager,
                         num_episodes=5)

        episodeStatistics = agent.train_model(gamma=0.5, epsilon=0.5, discountFactor=0.2)

        plotting.plot_episode_stats(episodeStatistics, smoothing_window=25)

        if args['test']:
            print("Testig")


if __name__ == "__main__":
    args = getParams()
    main(args)
