import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, Dense, Flatten, Conv1D
from keras.optimizers import Adam
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--learning_rate", type=float, help="Learning Rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    args = parser.parse_args()

    learning_rate = args.learning_rate
    epochs = args.epochs

    X_train = pd.read_csv("data/X_train.csv")
    X_val = pd.read_csv("data/X_val.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_val = pd.read_csv("data/y_val.csv")

    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    model = Sequential(
        [
            Conv1D(32, 2, activation="relu", input_shape=X_train[0].shape),
            BatchNormalization(),
            Dropout(0.2),
            Conv1D(64, 2, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        verbose=1,
    )

    os.makedirs("model", exist_ok=True)
    model.save("model/model.keras")


if __name__ == "__main__":
    main()
