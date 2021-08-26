try:
    import os
    import json
    import glob
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    from model_based_tfl_detection import test_find_tfl_lights
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image, ImageEnhance, ImageFilter


except ImportError:
    print("Need to fix the installation")
    raise


def main():
    """
    function that activates the other functions for creating a balanced dataset of tfl's and tfl's.
    """
    default_base_train = r"C:\Users\naorb\Desktop\Scale-up\projects\mobileye\leftImg8bit\leftImg8bit\train"
    default_base_val = r"C:\Users\naorb\Desktop\Scale-up\projects\mobileye\leftImg8bit\leftImg8bit\val"
    data_dir_train = r'gtFine2\train'
    data_dir_val = r'gtFine2\val'
    counter_result = write_dataset(default_base_train, data_dir_train, 2)
    write_dataset(default_base_val, data_dir_val, int(counter_result*(1/6)))
    print(f"You should now see the two binary files created in {data_dir_train} and "
          f"{data_dir_val} (data.bin and labels.bin).")


def balanced_quantity(tfl_images, no_tfl_images):
    """
    for balancing the number of images in the tfl's and the no-tfl's,
    this function uses data argumentation on the less and remove data from the surplus
    :param tfl_images: list of images which represents tfl
    :param no_tfl_images: list of images which represents stuff that aren't tfl
    :return: balanced lists
    """
    count = 0
    mirrored_tfl = []
    mirrored_no_tfl = []
    total_len = (len(tfl_images) + len(no_tfl_images)) * 0.4

    # use data argumentation trick

    while len(no_tfl_images) + len(mirrored_no_tfl) < total_len and count < len(no_tfl_images):
        mirrored_no_tfl.append(np.fliplr(no_tfl_images[count]))
        count += 1

    while len(tfl_images) + len(mirrored_tfl) < total_len and count < len(tfl_images):
        mirrored_tfl.append(np.fliplr(tfl_images[count]))
        count += 1

    tfl_images += mirrored_tfl
    no_tfl_images += mirrored_no_tfl

    # if that's not enough then remove some elements from the surplus

    while len(tfl_images) < (len(tfl_images) + len(no_tfl_images)) * 0.4:
        no_tfl_images.pop()

    while len(no_tfl_images) < (len(tfl_images) + len(no_tfl_images)) * 0.4:
        tfl_images.pop()

    return tfl_images, no_tfl_images


def write_dataset(default_base: str, data_dir: str, max_counter: int):
    """
    runs on all images in the cities found in the two paths (the leftImg8bit/leftImg8bit and gtFine/gtFine)
    and activate the test_tfl_detection for getting 2 balanced lists, one of tfl and no-tfl images
    and second of labels (True/False), both will be write to binary files, (data.bin and labels.bin)

    :param default_base: path to the directories represent cities,
     in those directories we can find the images which we want to build from them the dataset
    :param data_dir: path to the location where we want to place the dataset
    :param max_counter: the maximum quantity of images we want to put in the dataset
    :return: actual how many images are in the dataset (depends on the number of images in the directories)
    """
    counter = 1
    tfl_list = []
    no_tfl_list = []
    for filename in os.listdir(default_base):
        print(filename)
        flist = glob.glob(os.path.join(default_base, filename, '*_leftImg8bit.png'))
        for image in flist:
            label_fn = image.replace('_leftImg8bit.png', '_gtFine_labelIds.png').replace('leftImg8bit', 'gtFine', 2)
            print(counter)
            if not os.path.exists(label_fn):
                continue
            temp_tfl_list, temp_no_tfl_list = test_find_tfl_lights(image, label_fn)
            tfl_list += temp_tfl_list
            no_tfl_list += temp_no_tfl_list
            counter += 1

            if counter > max_counter:
                break
        if counter > max_counter:
            break
    tfl_list, no_tfl_list = balanced_quantity(tfl_list, no_tfl_list)
    labels = [1]*len(tfl_list) + [0]*len(no_tfl_list)
    data = tfl_list + no_tfl_list
    data = np.array(data, dtype=np.uint8)
    data.tofile(os.path.join(data_dir, 'data.bin'))
    labels = np.array(labels, dtype=np.uint8)
    labels.tofile(os.path.join(data_dir, 'labels.bin'))
    return counter


def load_data():
    """
    loads the dataset
    :return:
    """
    loaded_data = np.fromfile("gtFine/val/data.bin",  dtype=np.uint8)
    loaded_labels = np.fromfile("gtFine/train/labels.bin",  dtype=np.uint8)
    a = loaded_data.reshape(len(loaded_data)//19683, 81, 81, 3)
    for img in a:
        plt.imshow(img)
        plt.show()


def write_dataset_city(default_base: str, data_dir: str, max_counter: int):
    """
    runs on all images in the city found in the two paths (the leftImg8bit/leftImg8bit and gtFine/gtFine)
    and activate the test_tfl_detection for getting 2 balanced lists, one of tfl and no-tfl images
    and second of labels (True/False), both will be write to binary files, (data.bin and labels.bin)

    :param default_base: path to the images which we want to build from them the dataset
    :param data_dir: path to the location where we want to place the dataset
    :param max_counter: the maximum quantity of images we want to put in the dataset
    :return: actual how many images are in the dataset (depends on the number of images in the directories)
    """
    tfl_list = []
    no_tfl_list = []
    counter = 1
    flist = glob.glob(os.path.join(default_base, "aachen", '*_leftImg8bit.png'))
    for image in flist:
        label_fn = image.replace('_leftImg8bit.png', '_gtFine_labelIds.png').replace('leftImg8bit', 'gtFine', 2)
        print(counter)
        if not os.path.exists(label_fn):
            label_fn = None
        temp_tfl_list, temp_no_tfl_list = test_find_tfl_lights(image, label_fn)
        tfl_list += temp_tfl_list
        no_tfl_list += temp_no_tfl_list
        if counter > max_counter:
            break
        counter += 1
    tfl_list, no_tfl_list = balanced_quantity(tfl_list, no_tfl_list)
    labels = [1]*len(tfl_list) + [0]*len(no_tfl_list)
    data = tfl_list + no_tfl_list
    data = np.array(data, dtype=np.uint8)
    data.tofile(os.path.join(data_dir, 'data.bin'))
    labels = np.array(labels, dtype=np.uint8)
    labels.tofile(os.path.join(data_dir, 'labels.bin'))
    return counter


if __name__ == '__main__':
    main()
