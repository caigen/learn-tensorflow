"""
ref:
https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Activation(activation="relu", input_shape=[1]))
model.add(Dense(2, input_dim=1, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

print(model.summary())
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)