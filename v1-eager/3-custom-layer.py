"""
ref:
https://tensorflow.google.cn/tutorials/eager/custom_training
"""
import os
os.environ["KMP_WARNINGS"] = "0"

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

layer(tf.zeros([10, 5]))
print(layer)
print(layer.variables)
print(layer.kernel)
print(layer.bias)

print("====custom layers====")
class MyDenserLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenserLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape = [int(input_shape[-1]),
                                                 self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenserLayer(10)
print("layer:", layer(tf.zeros([10, 5])))
print("layer.trainable_variables:", layer.trainable_variables)

print("====composing layers====")
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=None):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])

print("====default sequential composing=====")
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2, 1,
                                                     padding="same"),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3, (1, 1)),
                              tf.keras.layers.BatchNormalization()])
print(my_seq(tf.zeros([1, 2, 3, 3])))