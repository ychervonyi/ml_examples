# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# About Adam http://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
import numpy as np

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# It does not make sense to normalize data because it is normalized inside the embedding layer
"""
def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

def min_value(inputlist):
    return min([sublist[-1] for sublist in inputlist])

print("Max value is ",max_value(X_train))
print("Min value is", min_value(X_train))

def norm(data):
    for row in range(len(data)):
        for col in range(len(data[row])):
            data[row][col] = data[row][col]/top_words
    return data

# X_train, X_test = norm(X_train), norm(X_test)

X_means = np.mean(X_train, axis=0)
X_stds = np.std(X_train, axis=0)
X_train = (X_train - X_means)/(X_stds+1e-6)
X_test = (X_test - X_means)/(X_stds+1e-6)

print("Data is normalized")
"""

# truncate and pad input sequences
max_review_length = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# weights0 = model.layers[0].get_weights()[0]
# weights1 = model.layers[1].get_weights()[0]
# weights2 = model.layers[2].get_weights()[0]
model.fit(X_train, y_train, nb_epoch=1, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
