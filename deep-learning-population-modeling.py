import pandas as pd
import numpy as np
import json
import datetime


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers, Input, initializers

class FixWeights(tf.keras.constraints.Constraint):

    def __call__(self, w):
        tf.keras.backend.set_value(w[0, 0], 0)
        tf.keras.backend.set_value(w[1, 1], 0)
        tf.keras.backend.set_value(w[2, 2], 0)
        tf.keras.backend.set_value(w[3, 3], 0)
 
        return w
    
with open("training_data.json", "r") as f:
    data_text = json.load(f)
data = {}
input_data = []
output_data = []
for v, new_v in data_text.items():
    # data[eval(v)] = eval(new_v)
    input_data.append(eval(v)) 
    output_data.append(eval(new_v)) 

# train_dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
# model = tf.keras.Sequential([
#   Input(shape=(1,)),
#   layers.Dense(4),
# ])
inputs = Input(shape=(4,))
outputs = layers.Dense(4,
    use_bias=False,
    kernel_constraint=FixWeights())(inputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit(train_dataset, epochs=5, callbacks=[tensorboard_callback])
model.fit(x=input_data, y=output_data, epochs=20, callbacks=[tensorboard_callback])

model.summary()

print(model(np.array([[0.25, 0.25, 0.25, 0.25]])))

print(model.trainable_variables) 