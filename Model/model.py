from Model import model_based_tfl_detection, SFM
from Model.SFM_standAlone import FrameContainer
from Model.neural_network_model import TFL_NeuralNetworkModel
import numpy as np


class Model:
    neural_network = TFL_NeuralNetworkModel()

    @staticmethod
    def detect_candidates(img: np.array):
        return model_based_tfl_detection.find_candidates(img)

    def filter_tfl(self, img: np.array, candidates: np.array, auxiliary: np.array):
        cropped_images = model_based_tfl_detection.cropped_images(img, candidates)
        boolean_list = self.neural_network.detect_traffic_lights(cropped_images)
        tfl_list = [candidates[i] for i in range(len(candidates)) if boolean_list[i] == 1]
        auxiliary = [auxiliary[i] for i in range(len(candidates)) if boolean_list[i] == 1]
        return tfl_list, auxiliary

    def calc_distances(self, prev_tfl, curr_tfl, em, focal, pp):
        # initial prev container
        prev_container = FrameContainer()
        prev_container.traffic_light = prev_tfl

        # initial current container
        curr_container = FrameContainer()
        curr_container.traffic_light = curr_tfl
        curr_container.EM = em

        # calculate distances
        curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
        if curr_container.traffic_lights_3d_location.any():
            return curr_container.traffic_lights_3d_location[:, 2].tolist()
