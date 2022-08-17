#!/usr/bin/env python3

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 25

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 201

def solution(input_layer=None):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """


    model = models.Sequential()
    # data augmentation
    model.add(layers.RandomFlip('horizontal'))
    model.add(layers.RandomRotation(0.25))
    model.add(layers.RandomZoom(0.25))
    # conv layers
    # use dropout for regularization
    model.add(layers.Conv2D(16, (5, 5), strides=2,
                            activation='relu', input_shape=(160, 160, 6)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(32, (5, 5), strides=2, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # fcnn
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # use softmax as model is predicting classes
    model.add(layers.Dense(5, activation='softmax'))

    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #           metrics=['accuracy'])

    return model

class CropImageLayer(tf.keras.layers.Layer):
    """
        keras Layer which crops and resizes input images to the highest 
        ranking bottle detection
    """
    def __init__(self, **kwargs):
        """
        Set bottle id and call init of super class
        """
        super().__init__(**kwargs)
        self.bottle_id = 44


    def call(self, input, boxes, scores, classes):
        """
        Crops and resizes input images such that they show the bottle detection
            with the highest detection score
        Parameters:
            input: input tensor of shape (batch_size, width, height, 3)
            boxes: np.array of shape (batch_size, classification_per_image, 4) containing
                coordinates of detection boxes
            classes: np.array of shape (batch_size, classification_per_image) containig
                class ids of the classifications
        Returns:
            inputimages respectively cropped to the best bottle detection box
        """
        boxes_filtered = []
        box_ind = []
        BOTTLE_ID = 44 # bottle id for object detection
        GLASS_ID = 46 # bottle id for object detection
        bottle_boxes_filtered, glass_boxes_filtered = [], []
        bottle_box_ind, glass_box_ind = [], []
        if not boxes.shape[0]:
            return tf.concat([input, input], axis = 3)

        for i in range(boxes.shape[0]): #batch
            for j in range(boxes.shape[1]): # detections
                # if classes[i][j] == BOTTLE_ID:
                    # print("(i,j):(" + str(i) + "," + str(j) + ")" + " score: " + str(scores[i][j].numpy()) + " class: " + str(classes[i][j].numpy()))
                if classes[i][j] == BOTTLE_ID and scores[i][j] > 0.5:
                    bottle_box = boxes[i][j].numpy() / 159
                    break
            # do not crop if no bottle was detected
            if j == boxes.shape[1] - 1:
                bottle_box = np.array([0, 0, 1, 1])
            bottle_box_ind.append(i)
            bottle_boxes_filtered.append(bottle_box)

        for i in range(boxes.shape[0]): #batch
            for j in range(boxes.shape[1]): # detections
                # if classes[i][j] == GLASS_ID:
                    # print("(i,j):(" + str(i) + "," + str(j) + ")" + " score: " + str(scores[i][j].numpy()) + " class: " + str(classes[i][j].numpy()))
                if classes[i][j] == GLASS_ID and scores[i][j] > 0.5:
                    glass_box = boxes[i][j].numpy() / 159
                    break
            # do not crop if no bottle was detected
            if j == boxes.shape[1] - 1:
                glass_box = np.array([0, 0, 1, 1])
            glass_box_ind.append(i)
            glass_boxes_filtered.append(glass_box)

        batch_glass_cropped = tf.image.crop_and_resize(input, glass_boxes_filtered, glass_box_ind, (160, 160))
        batch_bottle_cropped = tf.image.crop_and_resize(input, bottle_boxes_filtered, bottle_box_ind, (160, 160))
        batch_data_cropped = tf.concat([batch_glass_cropped, batch_bottle_cropped], axis = 3)
        return batch_data_cropped


class ComposedModel(tf.keras.Model):
    """
    Model composed of a pretrained object detector and a prediction head trained on the image dataset
    """
    def __init__(self):
        super(ComposedModel, self).__init__()
        # path to efficientdet model on tf hub
        model_handle = "https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1"
        # path of prediction head model weights
        prediction_head_path = "prediction_head_model"
        self.efficientdet = hub.KerasLayer(model_handle, trainable=False)
        self.crop_image_layer = CropImageLayer()
        self.prediction_head = tf.saved_model.load(prediction_head_path)


    def call(self, inputs):
        """
            Compute classification scores for the input batch
            Parameters:
                inputs: input tensor to be classified of shape (batch_size, height, width, 3)
            Return: softmax predictions of shape (batch_size, number_of_classes)
        """
        # inputs_uint8 = tf.cast(tf.math.scalar_mul(255.0, inputs, name=None), dtype=tf.uint8, name=None)
        # convert to uint8 from float32 as efficientdet expects image pixels in [0,256]
        inputs_uint8 = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=False)
        # perform object detection on input to obtain boxes and classes of classification candidates
        boxes, scores, classes, _ = self.efficientdet(inputs_uint8)
        # crop inputs respectively to bottles with highest classification score
        inputs_cropped = self.crop_image_layer(inputs, boxes, scores, classes)
        # compute predictions for respective image classes
        result = self.prediction_head(inputs_cropped)
        return result
