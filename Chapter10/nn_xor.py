from utilities.nn import NeuralNetwork
import numpy as np

# Construct the 'XOR' dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the NN
print('[INFO]: Training....')
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# Test the NN
print('[INFO]: Testing....')

# Loop over the data points
for (x, target) in zip(X, y):
    # Make a prediction and display the result
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print('[INFO]: Data={}, Ground Truth={}, Prediction={:.4f}, Step={}'.format(x, target[0], pred, step))

