#!/usr/bin/env python3

"""Train and evaluate the model

This file trains the model upon the training data and evaluates it with
the eval data.
It uses the arguments it got via the gcloud command.
"""

import os
import argparse
import logging
import base64

from datetime import datetime

from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np

from final_task import export_model
import model


def test_exported_model():
    """
    Compare the model directly after training to the exported model.

    Even though the models receive the same image (just in different formats) the predictions differ.
    The error therefore probably comes from the export of the model

    # """
    # create composed model, the solution to the coding challenge
    ml_model = model.ComposedModel()
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    basename = "composed_model_" + time_str
    # export to "output/composed_model"
    export_model(ml_model, "output", basename)
    # reload model which has just been exported
    ml_model_reloaded = tf.saved_model.load(os.path.join("output", basename))

    # open test image
    with open("0chimayblue_000.jpg", "rb") as img_file:
        # open test image encoded in base64
        img_base64 = base64.b64encode(img_file.read(), altchars=str.encode("-_"))
    # create image from base64 encoded string
    img = Image.open(BytesIO(base64.b64decode(img_base64, altchars=str.encode("-_"))))
    # from [0, 255] -> [0, 1]
    img = np.array(img) / 255.
    img = tf.expand_dims(img, axis=0)
    # convert base64 encoded image to string tensor
    img_str = tf.io.decode_base64(img_base64)
    img_str = tf.expand_dims(img_str, axis=0)
    predictions = ml_model(img, training=False)  # predictions for trained model
    predictions_reloaded = ml_model_reloaded(img_str, training=False)  # predictions for trained model which was exported and then reimported
    print("trained model: " + str(predictions.numpy())) #output: [0.01973824, 0.00156368, 0.3084035,  0.00962214, 0.6606724]
    print("served model: " + str(predictions_reloaded[1].numpy())) #output:  [0.00097238, 0.03858528, 0.00063064, 0.502958, 0.4568537]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)
    # run test
    test_exported_model()
