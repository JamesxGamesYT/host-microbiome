import pandas as pd
import numpy as np
import json
import datetime
import os
import matplotlib.pyplot as plt
from random import randint


print(os.getcwd())
print(os.listdir())


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers, Input, initializers, regularizers
from simulation import generate_training_data, runplot

# Cut all those tensorflow warnings
import logging
tf.get_logger().setLevel(logging.ERROR)
# sess = tf.Session()
# Custom constraint settings
class FixWeights(tf.keras.constraints.Constraint):

    def __call__(self, w):
        # tf.keras.backend.set_value(w[0, 0], 0)
        # tf.keras.backend.set_value(w[1, 1], 0)
        # tf.keras.backend.set_value(w[2, 2], 0)
        # tf.keras.backend.set_value(w[3, 3], 0)

        # tf.keras.backend.set_value(w[0, 1], 0)
        # tf.keras.backend.set_value(w[0, 2], 0)
        # tf.keras.backend.set_value(w[0, 3], 0)
        # tf.keras.backend.set_value(w[1, 2], 0)
        # tf.keras.backend.set_value(w[1, 3], 0)
        # tf.keras.backend.set_value(w[2, 3], 0)
        
        # Ensures columns sum to 1
        # tf.keras.backend.set_value(w[0,3], 1-w[0,0]-w[0,1]-w[0,2])
        # tf.keras.backend.set_value(w[1,3], 1-w[1,0]-w[1,1]-w[1,2])
        # tf.keras.backend.set_value(w[2,3], 1-w[2,0]-w[2,1]-w[2,2])
        # tf.keras.backend.set_value(w[3,3], 1-w[3,0]-w[3,1]-w[3,2])
        # tf.keras.backend.set_value(w[3,0], 1-w[2,0]-w[1,0]-w[0,0])
        # tf.keras.backend.set_value(w[3,1], 1-w[2,1]-w[1,1]-w[0,1])
        # tf.keras.backend.set_value(w[3,2], 1-w[2,2]-w[1,2]-w[0,2])
        # tf.keras.backend.set_value(w[3,3], 1-w[2,3]-w[1,3]-w[0,3])

        # Ensures columns sum to 1
        # x = tf.math.segment_sum(w, tf.constant([0, 0, 0, 1])).numpy()[0]
        # tf.keras.backend.set_value(w[3,0], 1-x[0])
        # tf.keras.backend.set_value(w[3,1], 1-x[1])
        # tf.keras.backend.set_value(w[3,2], 1-x[2])
        # tf.keras.backend.set_value(w[3,3], 1-x[3])
        return w

def prepare_data(load, W=None):
    '''
    Loads/creates training data for invasion modeling
    '''
    print("Prepare being called!", load)
    if load:
        with open("training_data/100_invasion_training_data.json", "r") as f:
            data_text = json.load(f)
    else:
        if W:
            print("W being passed!")
            data_text = generate_training_data(W, format="invasion", save=True)
        else:
            print("no W being passed!")
            data_text = generate_training_data(format="invasion", save=True)
    data = {}
    input_data = []
    output_data = []
    for v, new_v in data_text.items():
        # data[eval(v)] = eval(new_v)
        input_data.append(np.array(eval(v)))
        output_data.append(np.array(new_v))
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    return input_data, output_data

def plot_distribution(load):
    """
    Plot the distributions of training data
    """
    input_data, output_data = prepare_data(load)
    n = [x[1][1] for x in input_data]
    a = [x[1][0] for x in input_data]
    b = [x[0][0] for x in input_data] 
    c = [x[0][1] for x in input_data]
    print(len(n), "number of n")
    plt.ylim([0, 500])
    plt.hist(n, bins=100, alpha=0.5)
    plt.hist(a, bins=100, alpha=0.5)
    plt.hist(b, bins=100, alpha=0.5)
    plt.hist(c, bins=100, alpha=0.5)
    plt.savefig("graphs/distribution.png") 
    plt.clf()
    m = []
    for i in range(10000):
        init_conditions = np.random.rand(4)
        init_conditions /= sum(init_conditions)
        m.append(init_conditions[0])
    plt.hist(m, bins=100)
    plt.savefig("graphs/random.png") 
    plt.clf()

def run_model(load, W=None, index=None):
    """
    Train, fit, and save the model parameters
    """
    if W:
        input_data, output_data = prepare_data(load, W)
    else:
        print("no w run model, load=", load)
        input_data, output_data = prepare_data(load)
    # train_dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
    # model = tf.keras.Sequential([
    #   Input(shape=(1,)),
    #   layers.Dense(4),
    # ])
    inputs = Input(shape=(4,))
    outputs = layers.Dense(4,
        use_bias=False,
        kernel_constraint=FixWeights(),
        kernel_regularizer=None,
        # kernel_regularizer=regularizers.L1(l1=0.001),
    )(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss = tf.keras.losses.MeanSquaredError(),
                        optimizer = tf.keras.optimizers.Adam(), run_eagerly=True)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model.fit(train_dataset, epochs=5, callbacks=[tensorboard_callback])
    model.fit(x=input_data, y=output_data, epochs=10, callbacks=[tensorboard_callback])

    model.summary()

    print(model(np.array([[0.25, 0.25, 0.25, 0.25]])))

    if index:
        model.save(f'{index}/invasion_model')
    else:
        model.save('saved_models/cycling_invasion_model')
    numpy_variables = np.array(model.trainable_variables)
    print(numpy_variables)
    eigenvalues, eigenvectors = np.linalg.eig(numpy_variables)
    if index:
        with open(f"{index}/eig.txt", "w") as f:
            f.write("Eigenvalues: " + str(eigenvalues))
            f.write("\n")
            f.write("Eigenvectors: " + str(eigenvectors))
            f.write("\n")
            f.write(str(numpy_variables))
        with open(f"{index}/W.txt", "w") as f:
            f.write(str(W))

def modelarray(n):
    for index in range(76, n+76):
        if not os.path.isdir(f"./{index}"):
            os.mkdir(f"./{index}")
        W = [
            [0,0, randint(0, 20), randint(0,20)],
            [0,0, randint(0, 20), randint(0,20)],
            [randint(0,20), randint(0,20), 0, 0],
            [randint(0,20), randint(0,20), 0, 0],
        ]
        run_model(W=W, load=False, index=index)
        runplot(format="invasion", W=W, index=index)

if __name__ == "__main__":
    # run_model(load=True)
    run_model(load=False)
    # modelarray(100)