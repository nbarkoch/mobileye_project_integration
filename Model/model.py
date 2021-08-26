# import model_based_tfl_detection
# import neural_network_model
import numpy as np


def detect_candidates(img: np.array):
    return [[500, 500], [510, 500], [520, 500], [700, 500], [710, 500]], ["red", "red", "red", "green", "green"]


def filter_tfl(img: np.array, candidates: np.array, auxiliary: np.array):
    return [[500, 500], [510, 500], [710, 500]], ["red", "red", "green"]


def calc_distances(prev_tfl, curr_tfl, focal, pp):
    return [3.5, 7.7, 18.8]
