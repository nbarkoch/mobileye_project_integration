import numpy as np

from Model import model
from Model.model import Model
from View.view import View
import pickle
from PIL import Image
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Controller:
    def __init__(self, pls_path: str):
        with open(pls_path, 'r') as pls:
            play_list = pls.read().splitlines()
            pickle_path = play_list[0]
            first_index = int(play_list[1])
            images_path = play_list[2:]
            self.tfl_man = TFL_Manager(pickle_path, first_index)

        self.run(images_path)

    def run(self, images_path: list):
        for image_path in images_path:
            self.tfl_man.tfl_detection(image_path)


class TFL_Manager:
    data, curr_frame_id = None, None
    prev_img, prev_tfl, prev_ax = None, None, None
    pp, focal = None, None
    m = Model()

    def __init__(self, pkl_path: str, frame_id: int):
        self.curr_frame_id = frame_id
        with open(os.path.join(ROOT_DIR + r"\..", pkl_path), 'rb') as pklfile:
            self.data = pickle.load(pklfile, encoding='latin1')
            self.pp, self.focal = self.data["principle_point"], self.data["flx"]

    def tfl_detection(self, image_path):
        img = np.array(Image.open(os.path.join(ROOT_DIR + r"\..", image_path)))
        candidates, auxiliary = self.m.detect_candidates(img)

        curr_tfl, curr_ax = self.m.filter_tfl(img, candidates, auxiliary)

        EM = np.eye(4)
        if self.prev_img is not None:
            EM = np.dot(self.data['egomotion_' + str(self.curr_frame_id - 1) + '-' + str(self.curr_frame_id)], EM)
            distances = self.m.calc_distances(self.prev_tfl, curr_tfl, EM, self.focal, self.pp)
            if distances:
                View.write_lengths(curr_tfl, curr_ax, distances)
            View.draw_candidates(candidates, auxiliary)
            View.draw_traffic_lights(curr_tfl, curr_ax)
            View.show(img)

        self.prev_img = img
        self.prev_tfl = curr_tfl
        self.prev_ax = curr_ax
        self.curr_frame_id += 1


controller = Controller(r"..\play_list.pls")