try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    import cv2
    import scipy
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image, ImageEnhance, ImageFilter
    import matplotlib.pyplot as plt


except ImportError:
    print("Need to fix the installation")
    raise


def highlight_lights(image: np.ndarray):
    """
    turning an numpy array image into a black white image with white spots
    which are suspicious for being a traffic light, using masks, cv2 functions as blur and threshold
    :param image: The image itself as np.uint8, shape of (H, W, 3)
    :return: 2D black-white image with black background and white spots which are the tfl that detected
    """
    mask_red = np.all(image[:, :] <= [180, 235, 235], axis=-1)
    mask_green = np.all(image[:, :] <= [245, 235, 245], axis=-1)
    mask_blue = np.all(image[:, :] <= [245, 245, 190], axis=-1)

    mask_white = np.ma.mask_or(np.all(image[:, :] >= [190, 190, 190], axis=-1),
                               np.all(image[:, :] <= [155, 155, 155], axis=-1))

    mask = np.ma.mask_or(np.ma.mask_or(mask_red, mask_green), np.ma.mask_or(mask_blue, mask_white))
    image[mask] = [0, 0, 0]

    image[np.logical_not(mask)] = [255, 255, 255]

    # a better/worse way to detect the lights (get more tfl results = get more noises)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    fixed = cv2.threshold(blurred, 26, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (7, 7), 0)
    fixed = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (19, 19), 0)
    fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (19, 19), 0)
    fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

    # another way to detect the lights (get less tfl results = get less noises)
    # blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # fixed = cv2.threshold(blurred, 26, 255, cv2.THRESH_BINARY)[1]
    # blurred = cv2.GaussianBlur(fixed, (17, 17), 0)
    # fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

    return fixed[..., :3]  # return result 2D


def find_tfl_lights(c_image: np.ndarray):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = np.array(Image.open('kernel.png').convert('L'))
    kernel = kernel.astype('f')

    highlighted_image = highlight_lights(c_image.copy())
    fixed_image = \
        sg.convolve(np.dot(highlighted_image, [0.2125, 0.7154, 0.0721]), kernel, mode='same', method='auto')

    max_filter_image = scipy.ndimage.maximum_filter(fixed_image, 50)

    candidates = []
    auxiliary = []

    for i in range(len(max_filter_image)):
        for j in range(len(max_filter_image[i])):
            if max_filter_image[i][j] == fixed_image[i][j] and max_filter_image[i][j] > 1500000:
                candidates.append((i, j))
                if c_image[i][j][0] > c_image[i][j][1] >= c_image[i][j][2]:
                    auxiliary.append("red")
                else:
                    auxiliary.append("green")
    return candidates, auxiliary
