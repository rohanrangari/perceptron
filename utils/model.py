"""
author: Rohan Rangari
email: rohanrangari@gmail.com
"""
import numpy as np
import logging
from tqdm import tqdm


class Perceptron:
    def __init__(self, eta, epochs):
        """
        eta: Learning Rate
        epcohs: No of Epochs
        """
        self.weights = np.random.randn(3) * 1e-4
        logging.info(
            f"[INFO]: Initial weights before training: {self.weights} "
        )
        self.eta = eta
        self.epochs = epochs

    def activationFunction(self, inputs, weights):
        """Activation Definition

        Args:
            inputs : Inputs for the function
            weights : Weights

        Returns:
            1 or 0 : If the z value is greater than 0 then 1 else 0.
        """
        z = np.dot(inputs, weights)
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        """To train the model

        Args:
            X : Inputs for the  model
            y : Output
        """
        self.X = X
        self.y = y

        X_with_bias = np.c_[self.X, -np.ones(shape=(len(self.X), 1))]
        logging.info(f"[INFO]: X with bias:\n {X_with_bias} ")

        for epoch in tqdm(
            range(self.epochs), total=self.epochs, desc="training the model"
        ):
            logging.info(f"*" * 10)
            logging.info(f"[INFO]: for epoch: {epoch} ")
            logging.info(f"*" * 10)

            y_hat = self.activationFunction(X_with_bias, self.weights)
            logging.info(
                f"[INFO]: Predicted value after forward propagation: {y_hat} "
            )
            self.error = self.y - y_hat
            logging.info(f"[INFO]: Error:\n {self.error} ")
            self.weights = self.weights + self.eta * np.dot(
                X_with_bias.T, self.error
            )
            logging.info(
                f"[INFO]: Updated weights after epoch: {epoch+1}/{self.epochs}: {self.weights} "
            )
            logging.info(f"@" * 10)

    def predict(self, X):
        """Prediction for the given input

        Args:
            X : Input(s)

        """
        X_with_bias = np.c_[X, -np.ones(shape=(len(X), 1))]
        return self.activationFunction(X_with_bias, self.weights)

    def total_loss(self):
        """Calculate the total_loss"""
        total_loss = np.sum(self.error)
        logging.info(f"[INFO]: Total Loss : {total_loss} ")
        return total_loss
