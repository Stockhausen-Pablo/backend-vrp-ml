import csv

from argsConfig import getParams

import src.Tour.TourManager as tm
from src.Tour.Stop import Stop

def main(args):
    print("Please enter the datafile name")
    dataSetName = input("Enter name of file on Desktop:")
    tm.clear()
    with open('data/'+dataSetName+'.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            tm.addStop(Stop(float(row[2]), float(row[1])))
    print(tm.getLength())
    tm.calculateDistances()
    print(tm.getListOfStops())

if __name__ == "__main__":
    args = getParams()
    main(args)
