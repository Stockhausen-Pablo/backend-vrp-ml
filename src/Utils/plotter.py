import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

import numpy as np
import src.Tour.TourManager as tManager

plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125


def plt_color(stop_id):
    cols = []
    for id in stop_id:
        if id == 0:
            cols.append('red')
        else:
            cols.append('cyan')
    return cols


def plot_coordinates_with_coordinates_as_label():
    """
    Plots all coordinates with coordinates (longitude, latitude) as the scatter-point labels.
    """
    _, stop_id, x, y, _, _, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = plt_color(stop_id)

    plt.scatter(x, y, s=250, c=cols)

    for i_x, i_y in zip(x, y):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    plt.show()


def plot_coordinates_with_stopnr__as_label():
    """
    Plots all coordinates with stop number as the scatter-point labels.
    """
    _, stop_id, x, y, _, _, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = plt_color(stop_id)

    plt.scatter(x, y, s=250, c=cols)

    stop_nr = 0
    for i_x, i_y in zip(x, y):
        plt.text(i_x,
                 i_y,
                 stop_nr,
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontsize=24)
        stop_nr += 1

    plt.show()


def plot_tour_with_stopnr_as_label(tours):
    """
    Plots constructed tour with stop number as the scatter-point labels.
    """
    def connectPoints(x, y, p1, p2, col):
        x1, x2 = x[p1], x[p2]
        y1, y2 = y[p1], y[p2]
        line, = plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        line.set_color(col)

    _, stop_id, x, y, _, _, _, _ = zip(*[stop.getStop() for stop in tManager.stops])

    cols = plt_color(stop_id)

    plt.scatter(x, y, s=250, c=cols)

    colors = cm.rainbow(np.linspace(0, 1, len(tours)))
    colors_for_plt_legend = []

    tour_id = 0
    for tour, color in zip(tours, colors):
        colors_for_plt_legend.append(color)
        for idy, stop in enumerate(tour):
            next_stop = tour[(idy + 1) % len(tour)]
            connectPoints(x, y, stop.stopid, next_stop.stopid, color)
        tour_id += 1

    rows = [mpatches.Patch(color=tourColor) for tourColor in colors_for_plt_legend]
    text_label = ['tour {}'.format(i + 1) for i in range(len(tours))]

    plt.legend(rows,
               text_label,
               fontsize=11)

    stop_nr = 0
    for i_x, i_y in zip(x, y):
        plt.text(i_x,
                 i_y,
                 stop_nr,
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontsize=24)
        stop_nr += 1

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


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    """
    Plots all tracked stats of all episodes.
    """
    # --------------------
    # PLOTS EPISODE LENGTH OVER TIME
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        fig1.close()
    else:
        fig1.show()

    # --------------------
    # PLOTS EPISODE REWARD OVER TIME
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        fig2.close()
    else:
        fig2.show()

    # --------------------
    # PLOTS TIME STEPS PER EPISODE
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        fig3.close()
    else:
        fig3.show()

    # --------------------
    # PLOTS AVERAGE REWARD PER TIMESTEP
    fig4 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_J_avR)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Timestep")
    if noshow:
        fig4.close()
    else:
        fig4.show()

    # --------------------
    # PLOTS DISCOUNTED REWARD PER EPISODE
    fig5 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_G_t)
    plt.xlabel("Episode")
    plt.ylabel("Discounted Reward")
    plt.title("Discounted Reward per Episode")
    if noshow:
        fig5.close()
    else:
        fig5.show()

    # --------------------
    # PLOTS OPTIMAL POLICY REWARD PER EPISODE
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
