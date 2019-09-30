"""
ref:
https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model


def vgg_block(layer_in, n_filters, n_conv):
    # add n convolutional layers
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding="same", activation="relu")(layer_in)

    # add max pooling layer
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)

    return layer_in


def single_vgg_block():
    # define model input
    visible = Input(shape=(256, 256, 3))
    # add vgg module
    layer = vgg_block(visible, 64, 2)
    # create model
    model = Model(inputs=visible, outputs=layer)
    # summarize model
    print(model.summary())
    # plot model architecture
    plot_model(model, to_file="vgg_block.png", show_shapes=True, show_layer_names=True)


def multiple_vgg_blocks():
    # define model input
    visible = Input(shape=(256, 256, 3))
    # add vgg module
    layer = vgg_block(visible, 64, 2)
    # add vgg module
    layer = vgg_block(layer, 128, 2)
    # add vgg module
    layer = vgg_block(layer, 256, 4)
    # create model
    model = Model(inputs=visible, outputs=layer)
    # summarize model
    print(model.summary())
    # plot model architecture
    plot_model(model, to_file="multiple_vgg_blocks.png", show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    single_vgg_block()
    multiple_vgg_blocks()