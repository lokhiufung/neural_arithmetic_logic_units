import tensorflow as tf


# Neural Accumulator
class NAC(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(NAC, self).__init__()
    self.num_outputs = num_outputs
    
  def build(self, input_shape):
    self.W_hat = self.add_weight('W_hat', shape=[int(input_shape[-1]), self.num_outputs], trainable=True)
    self.M_hat = self.add_weight('M_hat', shape=[int(input_shape[-1]), self.num_outputs], trainable=True)
    
  def call(self, input_):
#     print(self.W_hat, self.M_hat)
    W = tf.nn.tanh(self.W_hat) * tf.nn.sigmoid(self.M_hat)
    return tf.matmul(input_, W)



