"""Containing all the utility methods"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")


def prepare_data(df):
    """To prepare input data and output data from the dataframe.

    Args:
        df (pandas.dataFrame): Dataset

    """

    X = df.drop(axis="columns", labels=["y"])
    y = df["y"]
    return X, y


def save_model(model, filename):
    """To save the trained model to a file.

    Args:
        model : Trained model
        filename : Filename for the trained model.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, filename)
    joblib.dump(model, file_path)


def save_plot(df, file_name, model):
    """To save the plot

    Args:
        df : Dataset
        file_name : Filename to be given for the plot
        model : Trained model
    """

    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_boundary(X, y, classifier, resolution=0.02):
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])

        X = X.values
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x1_min, x1_max, resolution),
        )
        Z = classifier.predict(np.array([xx1.ravel(), xx1.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()

    X, y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_boundary(X, y, model)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, file_name)
    joblib.dump(model, plot_path)
    plt.savefig(plot_path)
