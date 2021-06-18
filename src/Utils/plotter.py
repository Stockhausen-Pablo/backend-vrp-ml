import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

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


def plot_baseline_estimate(V, title="Baseline Estimate"):
    """
    Plots the baseline estimate as a surface plot.
    """
    states = list(V.keys())
    values = list(V.values())

    fig = plt.figure(figsize=(10, 5))
    plt.bar(range(len(states)), values, width=0.4)
    plt.xticks(range(len(states)), states)
    plt.xlabel("State Hashes")
    plt.ylabel("Current State-Value")
    plt.title(title)
    plt.show()


# ----------------------
# Inspired by: https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
# ----------------------
def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        fig1.close()
    else:
        fig1.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig2.close()
    else:
        fig2.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        fig3.close()
    else:
        fig3.show()

    # Plot average Reward per Timestep
    fig4 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_J_avR)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Timestep")
    if noshow:
        fig4.close()
    else:
        fig4.show()

    # Plot discounted Reward over timestep
    fig5 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_G_t)
    plt.xlabel("Episode")
    plt.ylabel("Discounted Reward")
    plt.title("Discounted Reward per Episode")
    if noshow:
        fig5.close()
    else:
        fig5.show()

    # Plot optimal policy reward over episode
    fig6 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_policy_reward)
    plt.xlabel("Episode")
    plt.ylabel("Optimal Policy Reward")
    plt.title("Optimal Policy Reward per Episode")
    if noshow:
        fig6.close()
    else:
        fig6.show()

    return fig1, fig2, fig3, fig4, fig5, fig6

