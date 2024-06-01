import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


def main():
    model = load_model("model/model.keras")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    y_pred = model.predict(X_test)
    y_pred = y_pred >= 0.5
    np.savetxt("data/y_pred.csv", y_pred, delimiter=",")

    cm = confusion_matrix(y_test, y_pred)
    print(
        "Recall metric in the testing dataset: ",
        cm[1, 1] / (cm[1, 0] + cm[1, 1]),
    )

    np.savetxt("data/confusion_matrix.csv", cm, delimiter=",")


if __name__ == "__main__":
    main()
