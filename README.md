I trained a model composed of four elements to classify images of beers by their labels.
* Convert input data from float32 tensor to uint8 tensor as object detction layer expects uint8 tensor
* Perform object detection to obtain boxes, class labels and scores of detected objects
* Crop to the highest rated detection of a bottle and a glass. Resize and concat the bottle and the glass image. Tensor shape therefore changes from (batch_size, width, height, 3) to (batch_size, width, height, 6)
* Prediction head. Consists of a CNN and a Feedforward network

Only the prediction head layer is trained.

The file test_exported_model.py performs the test between the model at the end of the training referred to as trained model and the restored model.
The model at the end of the training is recreated by restoring only the weights of the prediction head layer.
The restored model is created by exporting the complete composed model and restoring the model from this saved model.
When a prediction is performed the restored model and the trained model would be expected to produce the same results.
To perform the test one can run `python test_exported_model.py`. However the results vary significantly, for example for image 000.jpg from 0chimayblue from the evaluation dataset:
* `trained model: [[9.7456181e-01 2.0527570e-04 1.9158632e-02 1.0962408e-03 4.9780323e-03]]`
* `restored model: [[0.00636891 0.9207034  0.00868604 0.00923812 0.05500359]]`

When passing the test image into the model the trained model creates a detection box tf.tensor of shape (1, 100, 4), the restored model creates a detection box Tensor with shape=(None, 100, 4).  The respective outputs also get printed to the console.

Tested with Python 3.8.10 and Tensorflow 2.9.1.
