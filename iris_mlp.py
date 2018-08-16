import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras

# Prepare data
iris = sns.load_dataset("iris")
X = iris.values[:, 0:4]
y = iris.values[:, 4]

# Make test and train set
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=0)


################################
# Evaluate Keras Neural Network
################################

# Make ONE-HOT
def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)


l2_reg = 0.0

# dropouts = [0.2, 0.4, 0.6, 0.8, 1.0]
dropout = 1.0
hiddens = [1, 16, 128, 1024, 4096]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
accs, val_accs = [], []
losses, val_losses = [], []
for hidden in hiddens:
    model = Sequential()
    model.add(Dense(hidden,
                    input_shape=(4,),
                    activation="relu",
                    W_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=keras.optimizers.Adam(lr=0.0001))
    print(model.summary())
    # Actual modelling

    history = model.fit(train_X, train_y_ohe,
                        batch_size=16,
                        epochs=500,
                        verbose=1,
                        validation_data=(test_X, test_y_ohe))
    score = model.evaluate(test_X, test_y_ohe, verbose=0)
    print('Test loss with hidden size %d: %.4f' % (hidden, score[0]))
    print('Test accuracy with hidden size %d: %.4f' % (hidden, score[1]))
    accs.append(history.history['acc'])
    val_accs.append(history.history['val_acc'])

    losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

# summarize history for accuracy
for i in range(len(accs)):
    plt.plot(accs[i], '--', label='Train hidden size '+str(hiddens[i]), color=colors[i])
    plt.plot(val_accs[i], '-', label='Val hidden size '+str(hiddens[i]), color=colors[i])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epch')
plt.legend()

plt.figure()
for i in range(len(losses)):
    plt.plot(losses[i], '--', label='Train hidden size '+str(hiddens[i]), color=colors[i])
    plt.plot(val_losses[i], '-', label='Val hidden size '+str(hiddens[i]), color=colors[i])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()