"""
ref:
http://tensorfly.cn/tfdoc/get_started/basic_usage.html
"""

import tensorflow as tf

# ====operator====
# constant op
matrix1 = tf.constant([[3., 3.]])

matrix2 = tf.constant([[2.], 
                       [2.]])

# matmul op
product = tf.matmul(matrix1, matrix2)

# ====graph====
# default graph
sess = tf.Session()

# ====run====
result = sess.run(product)
print(result)

# ====close====
sess.close()

# auto close
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], 
                               [2.]])
        product = tf.matmul(matrix1, matrix2)
        print(sess.run(product))
