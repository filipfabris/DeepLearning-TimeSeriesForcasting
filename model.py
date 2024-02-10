import os
import numpy as np
import tensorflow as tf

WINDOW_SIZE = 100
TELESCOPE = 18


class model:
    def __init__(self, path):
        self.model_A = tf.keras.models.load_model(
            os.path.join(path, "SubmissionModel/Model_A")
        )
        self.model_B = tf.keras.models.load_model(
            os.path.join(path, "SubmissionModel/Model_B")
        )
        self.model_C = tf.keras.models.load_model(
            os.path.join(path, "SubmissionModel/Model_C")
        )
        self.model_D = tf.keras.models.load_model(
            os.path.join(path, "SubmissionModel/Model_D")
        )
        self.model_E = tf.keras.models.load_model(
            os.path.join(path, "SubmissionModel/Model_E")
        )
        self.model_F = tf.keras.models.load_model(
            os.path.join(path, "SubmissionModel/Model_F")
        )

    def predict(self, X, categories):
        X = X[:, X.shape[1] - WINDOW_SIZE :]
        # create out array of dimension (X.shape[0], TELESCOPE)
        out = np.zeros((X.shape[0], TELESCOPE))

        index = 0
        for time_serie in X:
            # transform the time serie in a (1, WINDOW_SIZE, 1) array
            time_serie = time_serie.reshape((1, WINDOW_SIZE, 1))
            # predict the next TELESCOPE values
            if categories[index] == "A":
                prediction = self.model_A.predict(time_serie)[:TELESCOPE]
            elif categories[index] == "B":
                prediction = self.model_B.predict(time_serie)
            elif categories[index] == "C":
                prediction = self.model_C.predict(time_serie)
            elif categories[index] == "D":
                prediction = self.model_D.predict(time_serie)
            elif categories[index] == "E":
                prediction = self.model_E.predict(time_serie)
            elif categories[index] == "F":
                prediction = self.model_F.predict(time_serie)

            out[index] = prediction[0, :TELESCOPE, 0]
            index += 1

        return out
