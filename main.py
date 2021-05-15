import csv
import numpy as np

from tf_agents.specs import array_spec

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
                Stop(int(row[0]), float(row[2]), float(row[1]), int(row[3]), bool(row[4].lower() in ['true'])))
    tManager.calculateDistances()


def main(args):
    # Input
    print("Please enter the datafile name")
    dataSet = input("Enter name of stops file on Desktop:")
    # amountVehicles = input("How many vehicles:")
    # capacityWeight = input("What is the capacityWeight:")
    # capacityVolume = input("What is the capacityVolume:")

    # -----
    tManager.clear()

    # Load Stop Data
    loadStopData(dataSet)

    # Plot Coordinates of input stops
    plotCoordinates()

    if args['train']:
        antColony = AntManager(
            stops=tManager.getListOfStops(),
            start_stop=tManager.getStop(0),
            vehicleCount=1,
            discountAlpha=1.1,
            discountBeta=1.1,
            pheromone_evaporation_coefficient=.70,
            pheromone_constant=30000,
            iterations=10
        )
        result = antColony.mainloop()
        environment = VRPEnvironment(
            states=tManager.getListOfStops(),
            actions=array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action'),
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
