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
    # teacher and student not just keras model, but my custom Model
    self.student = student
    self.num_class = num_class
    self.temperature = temperature
    self.lambda_const = lambda_const
    # just keras model
    self.teacher_model = self.ready_for_lecture(teacher.model)
    self.student = student
    self.student.model = self.ready_for_lecture(self.student.model)


  def ready_for_lecture(self, model):
    # Right now we assume model's last layer is just activation layer
    logits = model.layers[-2].output
    prob = Activation('softmax')(logits)
    logits_T = Lambda(lambda x : x / self.temperature)(logits)
    prob_T = Activation('softmax')(logits_T)
    output = concatenate([prob, prob_T])
    return keras.Model(model.input, output)

  def distill_loss(self, y_true, y_pred):
    y_true, y_true_soft = y_true[:, :self.num_class], y_true[:, self.num_class:]

    # why 2d instead of 1d? because of batch?
    y_pred, y_pred_soft = y_pred[:, :self.num_class], y_pred[:, self.num_class:]
    return self.lambda_const*logloss(y_true, y_pred) + logloss(y_true_soft, y_pred_soft)

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
    y_true_soft = y_true[:, self.num_class:]
    y_pred_soft = y_pred[:, self.num_class:]
    return logloss(y_true_soft, y_pred_soft)

  def preprocess_data(self, x_train, y_train):
    shape = y_train.shape
    y_train_new = np.ndarray((shape[0], shape[1]*2))
    for i in range(len(y_train)):
      y_train_new[i] = self.teacher_model.predict(np.expand_dims(x_train[i],axis=0))[0]
    return y_train_new


  def teach(self, x_train, y_train, batch_size=128, epochs=1, callbacks=[], validation_data=None):
    metrics=[self.accuracy,
             self.top_5_accuracy,
             self.categorical_crossentropy,
             self.soft_logloss]

    self.student.model.compile(
      optimizer='adam',
      #optimizer=keras.optimizers.SGD(lr=1e-1,momentum=0.9,nesterov=True),
      loss=self.distill_loss,
      #loss=lambda y_true, y_pred: self.distill_loss(y_true, y_pred),
      metrics=metrics)

    y_train_new = self.preprocess_data(x_train, y_train)
    if validation_data != None:
      vx, vy = validation_data
      validation_data = (vx, self.preprocess_data(vx, vy))

    return self.student.train(x_train, y_train_new, batch_size, epochs, callbacks, validation_data)
