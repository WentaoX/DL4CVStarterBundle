from utilities.nn.cnn import LeNet
from keras.utils import plot_model

# Initialize LeNet and then write the network architecture visualization grpah to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)
