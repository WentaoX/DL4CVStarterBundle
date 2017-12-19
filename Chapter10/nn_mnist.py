from utilities.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Load the MNIST dataset and apply min/max scaling to scale the pixel intensity values to the range [0, 1] (each
# image is represented as an 8x8 = 64-dim feature vector
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())
print('[INFO]: Samples={}, Dimension={}'.format(data.shape[0], data.shape[1]))

# Construct the training and testing splits
(train_x, test_x, train_y, test_y) = train_test_split(data, digits.target, test_size=0.25)

# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

# Train the network
print('[INFO]: Training....')
nn = NeuralNetwork([train_x.shape[1], 32, 16, 10])
print('[INFO]: {}'.format(nn))
nn.fit(train_x, train_y, epochs=1000)

# Test the network
print('[INFO]: Testing....')
predictions = nn.predict(test_x)
predictions = predictions.argmax(axis=1)
print(classification_report(test_y.argmax(axis=1), predictions))
