import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # Initialise the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # Apply the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # Insert a column of 1's as the last entry of the feature matrix
        # This allows us the treat the bias as a trainable parameter with the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # Loop over each individual data point
            for (x, target) in zip(X, y):
                # Pass the dot product of the input features and weight matrix through the step function
                pred = self.step(np.dot(x, self.W))

                # Only performs a weight update if our prediction does not match the target
                if pred != target:
                    # Calculate the error
                    error = pred - target

                    # Update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, add_bias=True):
        # Ensure our input is a matrix
        X = np.atleast_2d(X)

        # Check to see if the bias column should be added
        if add_bias:
            # Insert a column of 1's as the last entry in the feature matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # Pass the dot product of the input features and weight matrix through the step function
        return self.step(np.dot(X, self.W))
