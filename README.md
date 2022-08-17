The solution to the coding challenge consists of a model composed of an four elements:
* Convert input data from float32 tensor to uint8 tensor as object detction layer expects uint8 tensor
* Perform object detection to obtain boxes, class labels and scores of detected objects
* Crop to the highest rated detection of a bottle and a glass. Resize and concat the bottle and the glass image. Tensor shape therefore changes from (batch_size, width, height, 3) to (batch_size, width, height, 6)
* Prediction head. Consists of a CNN and a Feedforward network

Only the prediction head layer is trained.

The file test_exported_model.py performs the test between the model at the end of the training referred to as trained model and the served model.
The model at the end of the training is recreated by restoring only the weights of the prediction head layer.
The served model is created by exporting the complete composed model and restoring the model from this saved model.
When a prediction is performed the served model and the trained model would be expected to produce the same results.
To perform the test once can run `python test_exported_model.py`. However the results vary significantly, for example for image 000.jpg from 0chimayblue from the evaluation dataset:
* `trained model: [[9.7456181e-01 2.0527570e-04 1.9158632e-02 1.0962408e-03 4.9780323e-03]]`
* `served model: [[0.00516105 0.92184716 0.01040765 0.01153011 0.05105405]]`

This also explains the signifcant gap between performance on the test dataset and the validation set (not used for training). The model reaches an accuracy of about 85% on the validation dataset and an accuracy of ~50% on the test dataset. 

Tested with Python 3.8.10 and Tensorflow 2.9.1.