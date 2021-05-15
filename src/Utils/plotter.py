import matplotlib.pyplot as plt

import src.Tour.TourManager as tManager


def pltcolor(stopid):
    cols = []
    for id in stopid:
        if id == 0:
            cols.append('red')
        else:
            cols.append('blue')
    return cols


def plotCoordinates():
    stopid, x, y, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = pltcolor(stopid)

    plt.scatter(x, y, s=250, c=cols)

    for i_x, i_y in zip(x, y):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    plt.show()
