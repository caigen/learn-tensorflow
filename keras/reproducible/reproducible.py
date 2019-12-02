"""
ref:
https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
"""

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import random_normal
from sklearn.metrics import mean_squared_error


# fit MLP to dataset and print error
def fit_model(X, y):
    # design network
    model = Sequential()
    model.add(Dense(10, input_dim=1, kernel_initializer=random_normal(seed=1)))
    model.add(Dense(1, kernel_initializer=random_normal(seed=1)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    model.fit(X, y, epochs=100, batch_size=len(X), verbose=0)
    # forecast
    yhat = model.predict(X, verbose=0)
    print(mean_squared_error(y, yhat[:, 0]))


# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLP friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
# repeat experiment
repeats = 10
for _ in range(repeats):
    fit_model(X, y)
