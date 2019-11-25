"""
ref:
https://machinelearningmastery.com/save-load-keras-deep-learning-models/
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# fix random seed for reproducibility
numpy.random.seed(7)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, epochs=1, batch_size=10, verbose=0)
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model: Sequential = model_from_json(loaded_model_json)
# load weight into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
