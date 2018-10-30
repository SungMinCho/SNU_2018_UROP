import tensorflow as tf
from tensorflow import keras
import os

class Model():
  def __init__(self, fn_build_model, name, fn_compile, save_dir=None):
    self.name = name
    if save_dir == None:
      save_dir = "models/" + name
    self.save_dir = save_dir
    self.model = fn_build_model()
    self.fn_compile = fn_compile
    self.compile()

  def compile(self):
    self.fn_compile(self.model)

  def train(self, x_train, y_train, batch_size=128, epochs=1, callbacks=[], validation_data=None):
    return self.model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=validation_data,
              callbacks=callbacks)

  def save(self):
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    self.model.save_weights(os.path.join(self.save_dir, self.name+'.h5'))

  def load(self):
    path = os.path.join(self.save_dir, self.name+'.h5')
    if os.path.exists(path):
      self.model.load_weights(path)
