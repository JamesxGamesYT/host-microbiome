import numpy as np
import copy
import json
import matplotlib.pyplot as plt
"""
Run the discrete replicator equations for 'iter' timesteps. 

Output a timeseries of the state of the system along with a timeseries of the mean fitness.

"""
def runsim(v0:int, iterations:int, n:int, W:np.array, method="outer", proportionalnoise=0, constantnoise=0):
    v = v0 #BigFloat.(v0)  # The current system state
    data =  [[x] for x in v0]  # Timeseries of system states
    meanfitness = [0]

    # Neural network training data
    # Keys = one time step, values are second timestep
    training_data = {}

	
    for i in range(iterations):
        if method == "outer":

            # Calculate next gen vector
            π1 = np.ones(n)
            dot = np.outer(np.array([v]), np.array([v]))
            new_v = [sum(x) for x in np.multiply(W, dot)]
            # v = np.multiply(np.array([v]), np.matmul(W, np.array([v], ).T).T)[0]
            # v = np.multiply(W, dot) # * π1  # Recombination and fitness 
            
            meanfitness.append(sum(v))  # Record mean fitness

            # Warning: The order of the noise might matter. Idk
            new_v += np.random.rand(n) * constantnoise  # Add constant noise
            new_v += new_v * np.random.rand(n) * proportionalnoise # Add prop. noise
            # (If we change v and rand(1,n) in the above code to be zero in some entries then we can add noise to only a subset of entries.)
            new_v /= sum(new_v)  # Normalization
            # training_data[tuple(v[0].tolist())] = tuple(new_v)
            training_data[v.tolist()] = new_v.tolist()
            v = new_v
            for i in range(len(v0)):
                data[i].append(v[i]) # Add main sim data to timeseries
            

        # An alternative calculation method, based on the dot product.
        # I have not implemented noise into this one yet.
        if method == "dot":
            oldv = copy.deepcopy(v)
            for j in range(len(v)):
                v[j] = oldv[j]*np.dot(W[j:], oldv)  # Recombination and fitness
            
            meanfitness.append(sum(v))
            v /= sum(v)  # Normalization
            for i in range(v0):
                data[i].append(v[i]) # Add main sim data to timeseries
    return data, meanfitness, training_data



W = np.array([
    [0.0, 0.0, 1.0, 8.9],
    [0.0, 0.0, 6.0, 3.0],
    [15.5, 8.5, 0.0, 0.0],
    [4.5, 15.9, 0.0, 0.0]
])

def runplot():
    data2x2, meanfit2x2, _ = runsim([.25, .25, .25, .25], 1000, 4, W, constantnoise=0, proportionalnoise=0)
    plt_1 = plt.figure(figsize=(40, 15))
    for i, pop in enumerate(data2x2):
        plt.plot(pop)
        with open(f"{i}_.txt", "w") as f:
            f.write(str(pop))
    plt.savefig("data.png")

def generate_training_data():
    # Training data generation
    n = 10
    total_training_data = {}
    for i in range(n):
        init_conditions = np.random.rand(4)
        print(init_conditions)
        data2x2, meanfit2x2, training_data = runsim(init_conditions, 100, 4, W, constantnoise=0, proportionalnoise=0)
        total_training_data = {**total_training_data, **training_data}
        print(len(training_data), len(total_training_data))
        print(f"{i+1}/{n} complete {len(total_training_data)}", end="\r")
    with open("training_data.json", "w") as f:
        json.dump(total_training_data, f, indent=4)

generate_training_data()