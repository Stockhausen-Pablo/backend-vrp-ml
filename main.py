import csv

from argsConfig import getParams

import src.Tour.TourManager as tManager

from src.Tour.Stop import Stop
from src.Utils.plotter import plotCoordinates


def loadStopData(dataSet):
    with open('data/stops/'+dataSet+'.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            tManager.addStop(Stop(int(row[0]),float(row[2]), float(row[1]), int(row[3])))
    tManager.calculateDistances()


def main(args):
    # Input
    print("Please enter the datafile name")
    dataSet = input("Enter name of stops file on Desktop:")
    amountVehicles = input("How many vehicles:")
    capacityWeight = input("What is the capacityWeight:")
    capacityVolume = input("What is the capacityVolume:")

    #-----
    tManager.clear()

    #Load Stop Data
    loadStopData(dataSet)

    #Plot Coordinates of input stops
    plotCoordinates()


if __name__ == "__main__":
    args = getParams()
    main(args)
