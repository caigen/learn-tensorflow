"""
ref:
https://tensorflow.google.cn/tutorials/eager/eager_basics
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf

print("====v1-eager====")
# v2.0 default eagerly
# v1.0
tf.enable_eager_execution()
print(tf.executing_eagerly())

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
# print(tf.encode_base64("hello world"))

print(tf.square(2) + tf.square(3))

print("====Tensor Shape====")
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

print("====numpy compatibility====")
import numpy as np

ndarray = np.ones([3, 3])
print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And Numpy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

print("====Test memory====")
tensor_from_ndarray = tf.constant(ndarray)
print(ndarray)
ndarray = ndarray + 1
print(ndarray)
print(tensor_from_ndarray)

print("====GPU acceleration====")
x = tf.random.uniform([3, 3])
print(x)
print("Is there a GPU available: ")
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0: ")
print(x.device.endswith('GPU:0'))
print(x.device)

print("====Device====")
import time
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start

    print(" 10 loops: {:0.2f}ms".format(1000 * result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
        x = tf.random.uniform([1000, 1000])
        print(x.device)
        assert x.device.endswith("CPU:0")
        time_matmul(x)

# Force execution on GPU
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

print("====Dataset====")
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
print(ds_tensors)

import tempfile
_, file_name = tempfile.mkstemp()

with open(file_name, "w") as f:
    f.write(
"""Line 1
Line 2
Line 3
""")

ds_file = tf.data.TextLineDataset(file_name)
print(ds_file)
print("\nElements of ds_tensors:")
for x in ds_tensors:
    print(x)

print("\nElements in ds_file:")
for x in ds_file:
    print(x)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print("\nElements of ds_tensors:")
for x in ds_tensors:
    print(x)

print("\nElements in ds_file:")
for x in ds_file:
    print(x)