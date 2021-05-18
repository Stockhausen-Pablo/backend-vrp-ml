import csv
import numpy as np

from argsConfig import getParams
from train import train_model

import src.Tour.TourManager as tManager

from src.Tour.Stop import Stop
from src.Utils.plotter import plotCoordinates
from src.Mdp.VRPEnvironment import VRPEnvironment
from src.Aco.AntManager import AntManager


def loadStopData(dataSet):
    with open('data/stops/' + dataSet + '.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            tManager.addStop(
                Stop(float(row[0]), int(row[1]), float(row[3]), float(row[2]), int(row[4]), int(row[5])))
    tManager.calculateDistances()


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

    # Plot Coordinates of input stops
    plotCoordinates()

    if args['convert']:
        print("-Starting convert-")

    if args['train']:
        print("-Entered Training Mode-")
        print("-Starting up Ant Colony Optimization to get Probability Matrix-")
        antManager = AntManager(
            stops=tManager.getListOfStops(),
            start_stop=tManager.getStop(0),
            vehicleWeight=capacityWeight,
            vehicleVolume=capacityVolume,
            vehicleCount=amountVehicles,
            discountAlpha=1.1,
            discountBeta=1.1,
            pheromone_evaporation_coefficient=.70,
            pheromone_constant=1,
            iterations=500
        )
        result = antManager.runACO()
        environment = VRPEnvironment(
            states=tManager.getListOfStops(),
            actions=[0,1],
            probabilityMatrix=np.zeros((tManager.getLength(), tManager.getLength())),
            rewardFunction=0,
            microHub=tManager.getStop(0),
            discountFactor=0.2,
            capacityDemand=0,
            vehicles=0,
            vehicleCapacity=0)

        train_model(0,
                    start_epoch=0,
                    end_epoch=5,
                    )


if __name__ == "__main__":
    args = getParams()
    main(args)
