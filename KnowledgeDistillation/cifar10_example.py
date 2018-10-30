import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
from tensorflow.python.tools import freeze_graph
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from distill import Distill

def load_data():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  # shape = (B,32,32,3) /(B,1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  mean = np.mean(x_train, axis=(0,1,2,3))
  std = np.std(x_train, axis=(0,1,2,3))
  x_train = (x_train - mean)/std
  x_test = (x_test - mean)/std

  y_train = keras.utils.to_categorical(y_train,10)
  y_test= keras.utils.to_categorical(y_test,10)
  return x_train, y_train, x_test, y_test

def fn_compile(model):
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Got architecture and weights from https://github.com/geifmany/cifar-vgg
def build_teacher_model():
  model = Sequential()
  weight_decay = 0.0005

  model.add(Conv2D(64, (3, 3), padding='same',
                   input_shape=[32,32,3],kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))


  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))


  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(Dropout(0.5))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  return model

def build_student_model():
  model = Sequential()
  model.add(Conv2D(32, 3, input_shape=[32,32,3], padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  # (16,16,32)
  model.add(Conv2D(64, 3, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  # (8, 8, 64)
  model.add(Conv2D(128, 3, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  # (4, 4, 128)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  # Need to do seperate last activation layer for distill.
  model.add(Dense(10))
  model.add(Activation('softmax'))

  return model
