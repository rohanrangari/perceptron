import numpy as np


class Perceptron:
    def __init__(self, eta, epochs):
        """
        eta: Learning Rate
        epcohs: No of Epochs
        """
        self.weights = np.random.randn(3) * 1e-4
        print(f"[INFO]: Initial weights before training: {self.weights} ")
        self.eta = eta
        self.epochs = epochs

    def activationFunction(self, inputs, weights):
        z = np.dot(inputs, weights)
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        self.X = X
        self.y = y

        X_with_bias = np.c_[self.X, -np.ones(shape=(len(self.X), 1))]
        print(f"[INFO]: X with bias:\n {X_with_bias} ")

        for epoch in range(self.epochs):
            print(f"*" * 10)
            print(f"[INFO]: for epoch: {epoch} ")
            print(f"*" * 10)

            y_hat = self.activationFunction(X_with_bias, self.weights)
            print(
                f"[INFO]: Predicted value after forward propagation: {y_hat} "
            )
            self.error = self.y - y_hat
            print(f"[INFO]: Error:\n {self.error} ")
            self.weights = self.weights + self.eta * np.dot(
                X_with_bias.T, self.error
            )
            print(
                f"[INFO]: Updated weights after epoch: {epoch+1}/{self.epochs}: {self.weights} "
            )
            print(f"@" * 10)

    def predict(self, X):
        X_with_bias = np.c_[X, -np.ones(shape=(len(X), 1))]
        return self.activationFunction(X_with_bias, self.weights)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"[INFO]: Total Loss : {total_loss} ")
        return total_loss
