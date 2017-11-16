import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Method: Used to compute the sigmoid activation value
def sigmoid_activation(x):
    """
    :param x: Feature matrix
    :return: Predictions matrix
    """
    return 1.0 / (1 + np.exp(-x))


# Method: Used to obtain a set of predictions
def predict(x, w):
    """
    :param x: Feature matrix
    :param w: Weights matrix
    :return: Predictions matrix
    """
    # Take the dot product between the features and weight matrices to get the prediction matrix
    predictions = sigmoid_activation(x.dot(w))

    # Apply a step function threshold for the binary outputs
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0] = 1

    return predictions


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', required=False, type=float, default=100,
                help='# of epochs')
ap.add_argument('-a', '--alpha', required=False, type=float, default=0.01,
                help='learning rate')
args = vars(ap.parse_args())

# Generate a 2-class 2D classification problem with 1,000 data points
(x, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0]), 1)

# Insert a column of 1's as the last entry of the feature matrix (bias trick)
x = np.c_[x, np.ones((x.shape[0]))]

# Split the data into training (50%) and testing (50%)
(train_x, test_x, train_y, test_y) = train_test_split(x, y, test_size=0.5, random_state=42)

# Initialize the weight matrix and list of losses
print('[INFO]: Training....')
w = np.random.randn(x.shape[1], 1)
losses = []

# Loop for the number of epochs
for epoch in np.arange(0, args['epochs']):
    # Take the dot product between the features and weight matrices to get the predictions matrix
    predictions = sigmoid_activation(train_x.dot(w))

    # Compute the error between the predictions and true values
    error = predictions - train_y
    loss = np.sum(error ** 2)
    losses.append(loss)

    # Compute the gradient (dot product between the features and prediction errors)
    gradient = train_x.T.dot(error)

    # Update the weight matrix by 'nudging' it in the negative direction
    w += -args['alpha'] * gradient

    # Check to see if an update should be displayed
    if epoch == 0 or (epoch+1) % 5 == 0:
        print('[INFO]: epoch={}, loss={:.4f}'.format(int(epoch + 1), loss))


# Evaluate the model
print('[INFO]: Evaluating....')
predictions = predict(test_x, w)
print(classification_report(test_y, predictions))

# Plot the classification (test) data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(test_x[:, 0], test_x[:, 1], marker='o', c=test_y, s=30)

# Plot the loss over time
plt.style.use('ggplot')
plt.figure()
plt.title('Training Loss')
plt.plot(np.arange(0, args['epochs']), losses)
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
