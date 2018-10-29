import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Lambda, concatenate
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from model import Model
import numpy as np

# Reference
# https://github.com/TropComplique/knowledge-distillation-keras/blob/master/knowledge_distillation_for_mobilenet.ipynb
class Distill:
  def __init__(self, teacher, student, num_class, temperature=2.0, lambda_const=1):
    self.teacher = teacher
    self.student = student
    self.num_class = num_class
    self.temperature = temperature
    self.lambda_const = lambda_const

  def distill_loss(self, y_true, y_pred):
    y_true, logits = y_true[:, :self.num_class], y_true[:, self.num_class:]
    y_soft = tf.nn.softmax(logits/self.temperature)

    # why 2d instead of 1d? because of batch?
    y_pred, y_pred_soft = y_pred[:, :self.num_class], y_pred[:, self.num_class:]
    return self.lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

  def accuracy(self, y_true, y_pred):
    y_true = y_true[:, :self.num_class]
    y_pred = y_pred[:, :self.num_class]
    return categorical_accuracy(y_true, y_pred)

  def top_5_accuracy(self, y_true, y_pred):
    y_true = y_true[:, :self.num_class]
    y_pred = y_pred[:, :self.num_class]
    return top_k_categorical_accuracy(y_true, y_pred)

  def categorical_crossentropy(self, y_true, y_pred):
    y_true = y_true[:, :self.num_class]
    y_pred = y_pred[:, :self.num_class]
    return logloss(y_true, y_pred)

  def soft_logloss(self, y_true, y_pred):
    y_soft = y_true[:, self.num_class:]
    y_pred_soft = y_pred[:, self.num_class:]
    return logloss(y_soft, y_pred_soft)

  def preprocess_data(self, x_train, y_train):
    shape = y_train.shape
    y_train_new = np.ndarray((shape[0], shape[1]*2))
    for i in range(len(y_train)):
      logit = self.teacher.model.predict(np.expand_dims(x_train[i],axis=0))[0]
      y_train_new[i] = np.concatenate([y_train[i],logit])
      return y_train_new


  def teach(self, x_train, y_train, batch_size=128, epochs=1, callbacks=[], validation_data=None):
    # Right now, it assumes that the last layer of teacher and student is
    # just an activation layer rather than some layer with the activation inside.
    # Might need to make this more general later.

    model = self.student.model
    logits = model.layers[-2].output
    # i know we already have a softmax layer and that we could use that but just..
    prob = Activation('softmax')(logits)

    logits_T = Lambda(lambda x: x/self.temperature)(logits)
    prob_T = Activation('softmax')(logits_T)

    output = concatenate([prob, prob_T])
    model = keras.Model(model.input, output)

    metrics=[self.accuracy,
             self.top_5_accuracy,
             self.categorical_crossentropy,
             self.soft_logloss]
    #metrics=['accuracy']

    model.compile(
      #optimizer='adam',
      optimizer=keras.optimizers.SGD(lr=1e-1,momentum=0.9,nesterov=True),
      loss=lambda y_true, y_pred: self.distill_loss(y_true, y_pred),
      metrics=metrics)
    self.student.model = model

    y_train_new = self.preprocess_data(x_train, y_train)
    if validation_data != None:
      vx, vy = validation_data
      validation_data = (vx, self.preprocess_data(vx, vy))

    self.student.train(x_train, y_train_new, batch_size, epochs, callbacks, validation_data)
