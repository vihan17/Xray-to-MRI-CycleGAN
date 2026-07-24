from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

image_extensions = {'jpg', 'png', 'jpeg'}


def is_image(file):
    file_extensions = file.split('.')
    if len(file_extensions) < 2:
        return False
    return file_extensions[1] in image_extensions


def get_images_in_folder(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and is_image(f)]


def load_images_to_array(folder_path):
    images_paths = get_images_in_folder(folder_path)
    return [np.array(Image.open(path)) for path in images_paths]
