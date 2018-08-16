'''Trains a simple deep NN on the MNIST dataset.

Plot how accuracy depends on the dropout rate
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)

x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

dropouts = [0.2, 0.4, 0.6, 0.8, 1.0]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
dropouts = [0.2]
accs, val_accs = [], []
losses, val_losses = [], []
for i in range(len(dropouts)):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,))) # 2048
    model.add(Dropout(dropouts[i]))
    model.add(Dense(2048, activation='relu')) # extra
    model.add(Dense(10, activation='softmax'))
    # model.add(Dense(10, activation='softmax', input_shape=(784,)))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.01),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss with dropout %.2f : %.4f'%(dropouts[i], score[0]))
    print('Test accuracy with dropout %.2f : %.4f'%(dropouts[i], score[1]))
    accs.append(history.history['acc'])
    val_accs.append(history.history['val_acc'])

    losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

# summarize history for accuracy
for i in range(len(accs)):
    plt.plot(accs[i], '--', label='train dropout '+str(dropouts[i]), color=colors[i])
    plt.plot(val_accs[i], label='test dropout '+str(dropouts[i]), color=colors[i])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.figure()
# summarize history for loss
for i in range(len(losses)):
    plt.plot(losses[i], '--', label='train dropout '+str(dropouts[i]), color=colors[i])
    plt.plot(val_losses[i], label='test dropout '+str(dropouts[i]), color=colors[i])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.show()