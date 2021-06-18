import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

import numpy as np

import src.Tour.TourManager as tManager

plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125


def pltcolor(stopid):
    cols = []
    for id in stopid:
        if id == 0:
            cols.append('red')
        else:
            cols.append('cyan')
    return cols


def plotCoordinatesWithCoordinatesLabel():
    _, stopid, x, y, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = pltcolor(stopid)

    plt.scatter(x, y, s=250, c=cols)

    for i_x, i_y in zip(x, y):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    plt.show()


def plotCoordinatesWithStopNrLabel():
    _, stopid, x, y, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = pltcolor(stopid)

    plt.scatter(x, y, s=250, c=cols)

    stopnr = 0
    for i_x, i_y in zip(x, y):
        plt.text(i_x,
                 i_y,
                 stopnr,
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontsize=24)
        stopnr += 1

    plt.show()


def plotTourWithStopNrLabel(tours):
    def connectPoints(x, y, p1, p2, col):
        x1, x2 = x[p1], x[p2]
        y1, y2 = y[p1], y[p2]
        line, = plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        line.set_color(col)

    _, stopid, x, y, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = pltcolor(stopid)

    plt.scatter(x, y, s=250, c=cols)

    colors = cm.rainbow(np.linspace(0, 1, len(tours)))
    colorsForLegend = []

    tourid =0
    for tour, color in zip(tours, colors):
        colorsForLegend.append(color)
        for idy, stop in enumerate(tour):
            next_stop = tour[(idy + 1) % len(tour)]
            connectPoints(x, y, stop.stopid, next_stop.stopid, color)
        tourid += 1

    rows = [mpatches.Patch(color=tourColor) for tourColor in colorsForLegend]
    textLabel = ['tour {}'.format(i+1) for i in range(len(tours))]

    plt.legend(rows,
               textLabel,
               #loc=1,
               #bbox_to_anchor=(1.05, 1),
               #borderaxespad=0.,
               fontsize=11)

    stopnr = 0
    for i_x, i_y in zip(x, y):
        plt.text(i_x,
                 i_y,
                 stopnr,
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontsize=24)
        stopnr += 1

    plt.show()

