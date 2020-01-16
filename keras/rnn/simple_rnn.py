import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional

model = Sequential()
model.add(Embedding(40, 100))
model.add(SimpleRNN(2))
model.add(Dense(33, activation='sigmoid'))
model.summary()

model_LSTM = Sequential()
model_LSTM.add(Embedding(40, 100))
model_LSTM.add(LSTM(32))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.summary()

BiLSTM = Sequential()
BiLSTM.add(Embedding(40, 100))
BiLSTM.add(Bidirectional(LSTM(32)))
BiLSTM.add(Dense(1, activation='sigmoid'))
BiLSTM.summary()


model_GRU = Sequential()
model_GRU.add(Embedding(40, 100))
model_GRU.add(GRU(32))
model_GRU.add(Dense(1))
model_GRU.summary()

BiGRU = Sequential()
BiGRU.add(Embedding(40, 100))
BiGRU.add(Bidirectional(GRU(32)))
BiGRU.add(Dense(1, activation='sigmoid'))
BiGRU.summary()

