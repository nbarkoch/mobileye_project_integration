import numpy as np

from Model import model
from View.view import View
import pickle
from PIL import Image
import os

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
    EM = np.eye(4)

    def __init__(self, pkl_path: str, frame_id: int):
        self.curr_frame_id = frame_id
        with open(os.path.join("..", pkl_path), 'rb') as pklfile:
            self.data = pickle.load(pklfile, encoding='latin1')
            self.pp, self.focal = self.data["principle_point"], self.data["flx"]

    def tfl_detection(self, image_path):
        img = np.array(Image.open(os.path.join("..", image_path)))
        candidates, auxiliary = model.detect_candidates(img)
        View.draw_candidates(candidates, auxiliary)
        curr_tfl, curr_ax = model.filter_tfl(img, candidates, auxiliary)
        View.draw_traffic_lights(curr_tfl, curr_ax)
        if self.prev_img is not None:
            self.EM = np.dot(self.data['egomotion_' + str(self.curr_frame_id - 1) + '-' + str(self.curr_frame_id)], self.EM)
            distances = model.calc_distances(self.prev_tfl, curr_tfl, self.focal, self.pp)
            View.write_lengths(curr_tfl, curr_ax, distances)
            View.show(img)

        self.prev_img = img
        self.prev_tfl = curr_tfl
        self.prev_ax = curr_ax
        self.curr_frame_id += 1






controller = Controller(r"..\play_list.pls")