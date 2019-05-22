import tensorflow as tf 


# Neural arithmetic logic unit
class NALU(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(NALU, self).__init__()
    self.num_outputs = num_outputs
    
  def build(self, input_shape):
    self.W_hat = self.add_weight('W_hat', shape=[int(input_shape[-1]), self.num_outputs], trainable=True)
    self.M_hat = self.add_weight('M_hat', shape=[int(input_shape[-1]), self.num_outputs], trainable=True)
    self.G = self.add_weight('G', shape=[int(input_shape[-1]), self.num_outputs])
  
  def call(self, input_):
    W = tf.nn.tanh(self.W_hat) * tf.nn.sigmoid(self.M_hat)
    m = tf.exp(tf.matmul(tf.log(tf.abs(input_ + 1e-8)), W))
    g = tf.matmul(input_, self.G)
    return g * tf.matmul(input_, W) + (1 - g) * m


if __name__ == '__main__':
  import numpy as np
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]
  inputs = tf.keras.Input(shape=(2,))
  x = NALU(2)(inputs)
  output = NALU(1)(x)
  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01), loss='mse')
  model.fit(np.zeros((10, 2)).astype(np.float32), np.zeros((10, 1)).astype(np.float32), batch_size=8, epochs=1, callbacks=callbacks)
