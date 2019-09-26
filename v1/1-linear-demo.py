"""
ref:
http://tensorfly.cn/tfdoc/get_started/introduction.html
"""

import tensorflow as tf
import numpy as np

# shape: 2x100
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# linear model
# shape: 1
b = tf.Variable(tf.zeros([1]))
# shape: 1x2
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# mean square error
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# start
sess = tf.Session()
sess.run(init)

# fit
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
