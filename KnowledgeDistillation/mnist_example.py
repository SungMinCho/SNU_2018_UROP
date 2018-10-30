import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.python.tools import freeze_graph
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from distill import Distill

def load_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # shape = (B,28,28) /(B,)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train,10)
  y_test= keras.utils.to_categorical(y_test,10)
  return x_train, y_train, x_test, y_test

def fn_compile(model):
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

def build_teacher_model():
  model = Sequential()
  model.add(Flatten(input_shape=(28,28)))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(100, activation='relu'))
  #model.add(Dense(10, activation='softmax'))
  # Need to do this way for distill function's assumption.
  model.add(Dense(10))
  model.add(Activation('softmax'))

  return model

def build_student_model():
  model = Sequential()
  model.add(Flatten(input_shape=(28,28)))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(50, activation='relu'))
  #model.add(Dense(10, activation='softmax'))
  # Need to do this for distill.
  model.add(Dense(10))
  model.add(Activation('softmax'))

  return model

def train(model, xd, yd, xt, yt):
  tbCb = keras.callbacks.TensorBoard('Graph')

  model.fit(xd, yd,
            batch_size=128,
            epochs=1,
            validation_data=(xt,yt),
            callbacks=[tbCb])


def main():
  xd,yd,xt,yt = load_data()

  teacher = Model(build_teacher_model, "teacher", fn_compile)
  student = Model(build_student_model, "student", fn_compile)

  callbacks = [keras.callbacks.TensorBoard('logs')]
  teacher.train(xd, yd, 128, 1, callbacks, (xt,yt))
  student.train(xd, yd, 128, 1, callbacks, (xt,yt))

  teacher.save()
  student.save()

  student2 = Model(build_student_model, "student2", fn_compile)
  distill = Distill(teacher, student2, 10, 5.0, 0.07)
  distill.teach(xd, yd, 128, 1, callbacks, (xt,yt))


if __name__=='__main__':
  main()
