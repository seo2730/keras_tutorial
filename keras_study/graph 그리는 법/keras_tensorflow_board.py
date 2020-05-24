# 0.  Package to use
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd

np.random.seed(3)

# 1. Data set

# Trainging set & Test set
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# Separate Training set &  Verification set
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

# Data set processing
x_train = x_train.reshape(50000,784).astype('float32') / 255.0
x_val = x_val.reshape(10000,784).astype('float32') / 255.0
x_test = x_test.reshape(10000,784).astype('float32') / 255.0

# Choose Trainin set & Verification set
train_rand_idxs = np.random.choice(50000,700)
val_rand_idxs = np.random.choice(10000,300)
x_train = x_train[train_rand_idxs]
y_train = y_train[train_rand_idxs]
x_val = x_val[val_rand_idxs]
y_val = y_val[val_rand_idxs]

# Label data One-hot encoding prcessing
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# 2. Modeling
model = Sequential()
model.add(Dense(64, input_dim=28*28, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. Model learning Process Setting
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

# 4. Model learning
# tensorboard --logdir=~/Project/Keras/_writing/graph
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_train, y_train, nb_epoch=1000, batch_size=32,validation_data=(x_val,y_val),callbacks=[tb_hist])

# 5. Looking learning processing(Have to use jupyter notebook)
'''%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplot()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val loss')

acc_ax.plot(hist.history['acc'],'b',label='train acc')
acc_ax.plot(hist.history['val_acc'],'g',label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
loss_ax.legend(loc='lower left')

plt.show()'''