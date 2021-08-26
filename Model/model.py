# import model_based_tfl_detection
# import neural_network_model
import numpy as np
from Model import SFM


def detect_candidates(img: np.array):
    return [[500, 500], [510, 500], [520, 500], [700, 500], [710, 500]], ["red", "red", "red", "green", "green"]


def filter_tfl(img: np.array, candidates: np.array, auxiliary: np.array):
    return [[500, 500], [510, 500], [710, 500]], ["red", "red", "green"]


def calc_distances(prev_tfl, curr_tfl, em, focal, pp):
    # initial prev container
    prev_container = SFM.FrameContainer()
    prev_container.traffic_light = prev_tfl

    # initial current container
    curr_container = SFM.FrameContainer()
    curr_container.traffic_light = curr_tfl
    curr_container.EM = em

    # calculate distances
    curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
    distances = []
    for i in range(len(curr_container.traffic_light)):
        # Next line is in a note because we want keep the order of lights.
        # if curr_container.valid[i]:
        distances.append(curr_container.traffic_lights_3d_location[i, 2])

    return distances
