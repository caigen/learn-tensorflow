"""
ref:
http://tensorfly.cn/tfdoc/get_started/os_setup.html
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()

print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)

print(sess.run(a + b))
