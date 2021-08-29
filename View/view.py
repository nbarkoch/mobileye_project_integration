import matplotlib.pyplot as plt
import numpy as np


class View:
    plt1, plt2, plt3 = None, None, None

    @classmethod
    def draw_candidates(cls, candidates, auxiliary):
        cls.plt1 = plt.subplot(311)
        make_plots(cls.plt1, candidates, auxiliary)

    @classmethod
    def draw_traffic_lights(cls, candidates, auxiliary):
        cls.plt2 = plt.subplot(312)
        make_plots(cls.plt2, candidates, auxiliary)

    @classmethod
    def write_lengths(cls, candidates, auxiliary, lengths, valid_list):
        cls.plt3 = plt.subplot(313)
        make_plots(cls.plt3, candidates, auxiliary)
        add_length(cls.plt3, candidates, lengths, valid_list)

    @classmethod
    def show(cls, image):
        cls.plt1.imshow(image)
        cls.plt2.imshow(image)
        cls.plt3.imshow(image)
        plt.show()


def make_plots(subplot, candidates, auxiliary):
    red_candidates = np.array(list(filter(lambda p: auxiliary[candidates.index(p)] == "red", candidates)))
    green_candidates = np.array(list(filter(lambda p: auxiliary[candidates.index(p)] == "green", candidates)))

    if red_candidates.any():
        subplot.plot(red_candidates[:, 0], red_candidates[:, 1], 'rx', markersize=4)
    if green_candidates.any():
        subplot.plot(green_candidates[:, 0], green_candidates[:, 1], 'g+', markersize=4)


def add_length(subplot, candidates, lengths, valid_list):
    for i in range(len(lengths)):
        if valid_list[i]:
            subplot.text(candidates[i][0], candidates[i][1], r'{0:.1f}'.format(lengths[i]), color='y')
