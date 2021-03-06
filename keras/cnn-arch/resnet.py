"""
ref:
https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model
from keras.layers import add


def residual_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of fitlers needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1, 1), padding="same",
                             activation="relu", kernel_initializer="he_normal",
                             )(layer_in)
    # conv1
    conv1 = Conv2D(n_filters, (3, 3), padding="same", activation="relu",
                   kernel_initializer="he_normal")(layer_in)
    # conv2
    conv2 = Conv2D(n_filters, (3, 3), padding="same", activation="linear",
                   kernel_initializer="he_normal")(conv1)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    layer_out = Activation("relu")(layer_out)
    return layer_out


# define model input
visible = Input(shape=(256, 256, 3))
# add resnet module
layer = residual_module(visible, 64)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file="residual_module.png")