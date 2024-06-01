import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import zipfile


def load_data(name):
    df = pd.read_csv(name)
    return df


def normalize_data(df):
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    return df


def split_data(df):
    X = df.iloc[:, df.columns != "Class"]
    y = df.iloc[:, df.columns == "Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=0
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(df, X_train, X_val, X_test, y_train, y_val, y_test):
    df.to_csv("data/creditcard.csv", index=False)
    X_train.to_csv("data/X_train.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_val.to_csv("data/y_val.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)


def main():
    os.makedirs("data", exist_ok=True)
    os.system("rm -rf data/*")

    with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall()

    df = load_data("creditcard.csv")
    df = normalize_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    save_data(df, X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
