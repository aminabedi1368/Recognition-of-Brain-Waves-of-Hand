from tensorflow import keras

import numpy

import time  # for benchmarking performance speed
from typing import Any


class App:

    #examples of model_type: A_3
    def __init__(self, model_type):

        self.model_type = model_type
        conv_type = model_type.split("_")[1] # TODO

        # find the model file
        if model_type == "B_3":
            self.model_file = 'model_patient_B.h5'

        # setting the number of channels
        if ("A" in model_type):
            num_EEG_channels = 62
            signal_duration = 100
        elif ("B" in model_type):
            num_EEG_channels = 62
            signal_duration = 100
        elif ("C" in model_type):
            num_EEG_channels = 31
            signal_duration = 100
        elif ("D" in model_type):
            num_EEG_channels = 46
            signal_duration = 100
        elif ("E" in model_type):
            num_EEG_channels = 10
            signal_duration = 75
        else:
            print("patient unknown")

        # input image dimensions
        if conv_type == '2' or conv_type == '3' or conv_type == '4':
            self.img_rows = num_EEG_channels
            self.img_cols = signal_duration
            self.num_image_channels = 1
        else:
            self.img_rows = 1
            self.img_cols = signal_duration
            self.num_image_channels = num_EEG_channels

        print(self.img_rows)

    # This method loads the data abd run the prediction
    def run(self, inputFile):

        print('started running')

        X, input_shape = self.loadData(inputFile)

        return self.run_model(X)

    # input: separat feetures and classes files
    # classes: 1 (non-mvement), 2 (left), 3 (right), but remove 1 for comparison -> 1868 in total, and 1109 without the "rest" class (same as the paper)
    # can work with or withour the "rest" class label
    # 1 = 759   2 = 546   3 = 563
    # output should be a 1 image channel, 46 X 100 images (46 electrode channels, 100 timesteps)
    def loadData(self, inputFile):

        print('formatting data')

        # the data
        X = numpy.transpose(numpy.loadtxt(inputFile, delimiter=","))

        print(keras.backend.image_data_format())
        if keras.backend.image_data_format() == 'channels_first':
            X = X.reshape(1, self.num_image_channels, self.img_rows, self.img_cols)
            input_shape = (self.num_image_channels, self.img_rows, self.img_cols)
        else:  # channels_last
            X = X.reshape(1, self.img_rows, self.img_cols, self.num_image_channels)
            input_shape = (self.img_rows, self.img_cols, self.num_image_channels)

        return X, input_shape

    def run_model(self, X_test):

        print('predicting')

        model = keras.models.load_model(self.model_file)

        return model.predict(X_test)


# =======running the app========

# read the input JSON
model_type = 'B_3'
inputFile = "patient_B_features.csv"

# run prediction
startt = time.time()

app = App(model_type)
prediction = app.run(inputFile)

print(prediction[0])

if prediction[0] < 0:
    result = 'L'
else:
    result = 'R'
print(result)

endt = time.time()
runTime = endt-startt
print(runTime)

# write to output JSON


