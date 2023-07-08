import numpy as np
import copy
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf

dir = os.path.dirname(os.path.abspath(__file__))

population_mapping = {
    0 : "A",
    1 : "B",
    2 : "a",
    3 : "b"
}
invasion_mapping = {
    0 : "Aa",
    1 : "Ab",
    2 : "Ba",
    3 : "Bb"
}

"""
Run the discrete replicator equations for 'iter' timesteps. 

Output a timeseries of the state of the system along with a timeseries of the mean fitness.

"""
def run_sim(v0:int, iterations:int, n:int, W:np.array, proportionalnoise=0, constantnoise=0, format="population", training=False):
    v = v0 #BigFloat.(v0)  # The current system state
    data =  [[] for x in range(n)]  # Timeseries of system states
    meanfitness = [0]

    # Neural network training data
    # Keys = one time step, values are second timestep
    training_data = {}
    previous_reformatted_dot = None
	
    for i in range(iterations):
        # Calculate next gen vector
        π1 = np.ones(n)
        dot = np.outer(np.array([v]), np.array([v]))
        new_v = [sum(x) for x in np.multiply(W, dot)]
        # v = np.multiply(np.array([v]), np.matmul(W, np.array([v], ).T).T)[0]
        # v = np.multiply(W, dot) # * π1  # Recombination and fitness 
        

        # Warning: The order of the noise might matter. Idk
        new_v += np.random.rand(n) * constantnoise  # Add constant noise
        new_v += new_v * np.random.rand(n) * proportionalnoise # Add prop. noise
        # (If we change v and rand(1,n) in the above code to be zero in some entries then we can add noise to only a subset of entries.)
        new_v /= sum(new_v)  # Normalization

        # Find the frequencies of the holobiont combinations
        reformatted_dot = np.array([dot[0][2]+dot[2][0], dot[0][3]+dot[3][0], dot[1][2]+dot[2][1], dot[1][3]+dot[3][1]])
        reformatted_dot /= sum(reformatted_dot)
        if training == "population":
            training_data[str(v.tolist())] = tuple(new_v)
            meanfitness.append(sum(new_v))  # Record mean fitness
        elif training == "invasion":
            if i >= 1:
                training_data[str(previous_reformatted_dot.tolist())] = reformatted_dot.tolist()
                meanfitness.append(sum(reformatted_dot))  # Record mean fitness

        previous_reformatted_dot = reformatted_dot
        v = new_v

        for i in range(len(v0)):
            if format == "population":
                data[i].append(v[i]) # Add main sim data to timeseries
            elif format == "invasion":
                data[i].append(reformatted_dot[i]) # Add main sim data to timeseries

    return data, meanfitness, training_data

def run_smeared_sim(v0:list, iterations:int, n:int, proportionalnoise=0, constantnoise=0):
    smear = [[ 0.941, -0.094,  0.148,  0.001],
       [ 0.121,  0.971, -0.016, -0.073],
       [-0.039, -0.002,  0.977,  0.065],
       [-0.002,  0.053, -0.039,  0.988]]
    model = tf.keras.models.load_model('./saved_models/cycling_invasion_model/')
    data = [[] for x in range(n)]
    meanfitness = [0]
    v = v0
    for i in range(iterations):
        print(i)
        # new_v = np.matmul(smear, v)
        new_v = model.predict(np.array([v]))[0]
        meanfitness.append(sum(new_v))
        new_v /= sum(new_v)
        v = new_v
        for i, val in enumerate(new_v):
            data[i].append(val)
    return data, meanfitness

# An array that cycles
W = np.array([
    [0.0, 0.0, 1.0, 8.9],
    [0.0, 0.0, 6.0, 3.0],
    [15.5, 8.5, 0.0, 0.0],
    [4.5, 15.9, 0.0, 0.0]
])

# An array that doesn't cycle
# W = np.array([
#     [0.0, 0.0, 5.0, 3.9],
#     [0.0, 0.0, 2.0, 8.0],
#     [15.5, 8.5, 0.0, 0.0],
#     [4.5, 15.9, 0.0, 0.0]
# ])


# W = np.array([
#     [0.0, 0.0, 9.0, 4.0],
#     [0.0, 0.0, 7.0, 13.0],
#     [11.5, 4.5, 0.0, 0.0],
#     [8.5, 17.9, 0.0, 0.0]
# ])

# W = np.array([
#     [0.0, 0.0, 5.0, 4.0],
#     [0.0, 0.0, 7.0, 8.0],
#     [15.5, 4.5, 0.0, 0.0],
#     [8.5, 15.9, 0.0, 0.0]
# ])

# W = np.array([
#     [0.0, 0.0, 5.0, 1.0],
#     [0.0, 0.0, 7.0, 8.0],
#     [15.5, 4.5, 0.0, 0.0],
#     [8.5, 15.9, 0.0, 0.0]
# ])

def runplot(format):
    data2x2, meanfit2x2, _ = run_sim([.25, .25, .25, .25], 200, 4, W, format=format, constantnoise=0, proportionalnoise=0)
    plt_1 = plt.figure(figsize=(40, 15))
    if format == "population":
        for i, pop in enumerate(data2x2):
            plt.plot(pop, label=population_mapping[i])
            with open(f"frequency_logs/{population_mapping[i]}_.txt", "w") as f:
                f.write(str(pop))
        plt.legend(loc="lower right")
        plt.savefig("graphs/population_data.png")
    elif format == "invasion":
        for i, pop in enumerate(data2x2):
            plt.plot(pop, label=invasion_mapping[i])
            with open(f"frequency_logs/{invasion_mapping[i]}_.txt", "w") as f:
                f.write(str(pop))
        data, meanfitness = run_smeared_sim([.25, .25, .25, .25], 200, 4)
        for i, pop in enumerate(data):
            plt.plot(pop, label="modeled_"+invasion_mapping[i])
            # with open(f"frequency_logs/{invasion_mapping[i]}_.txt", "w") as f:
                # f.write(str(pop))
        plt.legend(loc="lower right")
        plt.savefig("graphs/cycling_invasion_data.png")

def generate_training_data(format, save=False):
    # Training data generation
    n = 1000
    timesteps = 100
    total_training_data = {}
    for i in range(n):
        init_conditions = np.random.rand(4)
        data2x2, meanfit2x2, training_data = run_sim(init_conditions, timesteps, 4, W, constantnoise=0, proportionalnoise=0, training=format)
        total_training_data = {**total_training_data, **training_data}
        print(f"{i+1}/{n} complete {len(total_training_data)}", end="\r")
    if save:
        with open(f"{dir}/training_data/{timesteps}_{format}_training_data.json", "w") as f:
            json.dump(total_training_data, f, indent=4) 
        return total_training_data
    else:
        return total_training_data

if __name__ == "__main__":
    runplot("invasion")
    # generate_training_data("invasion", save=True)