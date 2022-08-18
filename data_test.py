#!/usr/bin/env python3

"""Load image files and labels

This file contains the method that creates data and labels from a directory.
"""
import os
from pathlib import Path

import numpy as np
import cv2

import tensorflow as tf

def create_data_with_labels(dataset_dir):
    """Gets numpy data and label array from images that are in the folders
    that are in the folder which was given as a parameter. The folders
    that are in that folder are identified by the beers they represent and
    the folder name starts with the label.

    Parameters:
        dataset_dir: A string specifying the directory of a dataset
    Returns:
        data: A numpy array containing the images
        labels: A numpy array containing labels corresponding to the images
    """
    image_paths_per_label = collect_paths_to_files(dataset_dir)

    images = []
    labels = []
    for label, image_paths in image_paths_per_label.items():
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
    data = np.array([preprocess_image(image.astype(np.float32))
                     for image in images])
#    data = tf.keras.utils.normalize(data, axis=-1, order=2)
    labels = np.array(labels)

    return data, labels

def collect_paths_to_files(dataset_dir):
    """Returns a dict with labels for each subdirectory of the given directory
    as keys and lists of the subdirectory's contents as values.

    Parameters:
        dataset_dir: A string containing the path to a directory containing
            subdirectories to different classes.
    Returns:
        image_paths_per_label: A dict with labels as keys and lists of file
        paths as values.
    """
    dataset_dir = Path(dataset_dir)
    beer_dirs = [f for f in sorted(os.listdir(dataset_dir)) if not f.startswith('.')]
    image_paths_per_label = {
        label: [
            dataset_dir / beer_dir / '{0}'.format(f)
            for f in os.listdir(dataset_dir / beer_dir) if not f.startswith('.')
        ]
        for label, beer_dir in enumerate(beer_dirs)
    }
    return image_paths_per_label

def preprocess_image(image):
    """Returns a preprocessed image.

    Parameters:
        image: A RGB image with pixel values in range [0, 255].
    Returns
        image: The preprocessed image.
    """
    image = image / 255.
    # r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # gray = np.expand_dims(gray, 2)
    # print(gray.shape)
    return image
