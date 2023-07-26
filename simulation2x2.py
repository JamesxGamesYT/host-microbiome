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
# invasion_mapping = {
#     0 : r"P_{Aa}",
#     1 : r"P_{Ab}",
#     2 : r"P_{Ba}",
#     3 : r"P_{Bb}"
# }

plt.rcParams.update({'font.size': 32})
# plt.rcParams.update({'text.usetex': True})
# plt.rcParams["font.family"] = "Times New Roman"


# An array that cycles
# W = np.array([
#     [0.0, 0.0, 1.0, 8.9],
#     [0.0, 0.0, 6.0, 3.0],
#     [15.5, 8.5, 0.0, 0.0],
#     [4.5, 15.9, 0.0, 0.0]
# ])

# Non-correspondence between modeled equilibrium and cycling dominance durations (index 169)
# W = np.array([[0, 0, 13, 11], [0, 0, 10, 16], [20, 1, 0, 0], [18, 1, 0, 0]])

# Index 151
# W = np.array([[0, 0, 14, 6], [0, 0, 16, 5], [9, 18, 0, 0], [3, 19, 0, 0]])

# Index 172, modeled divergence despite cycling
# W = np.array([
#     [0.0, 0.0, 12, 2],
#     [0.0, 0.0, 11, 20],
#     [6, 20, 0.0, 0.0],
#     [7, 14, 0.0, 0.0]
# ])

# index 168
W = np.array([[0, 0, 17, 18], [0, 0, 7, 14], [12, 2, 0, 0], [12, 20, 0, 0]])

# Index 161, tests for reliability of cycle length dominance
# W = np.array([[0, 0, 9, 13], [0, 0, 1, 16], [1, 18, 0, 0], [11, 12, 0, 0]])

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


# W = np.array([
#     [0.0, 0.0, 5.8, 6.1],
#     [0.0, 0.0, 14.3, 9.5],
#     [8.0, 10.1, 0.0, 0.0],
#     [11.5, 6.7, 0.0, 0.0]
# ])

# W = np.array([
#     [0.0, 0.0, 2, 10],
#     [0.0, 0.0, 9, 8],
#     [11, 14, 0.0, 0.0],
#     [20, 8, 0.0, 0.0]
# ])

# Index 93
# W = np.array([[0, 0, 19, 5], [0, 0, 10, 5], [13, 9, 0, 0], [17, 16, 0, 0]])

# Index 164
W = np.array([[0, 0, 13, 15], [0, 0, 10, 11], [6, 17, 0, 0], [6, 10, 0, 0]])

# Non cycling where it depends on initial conditions (index 89)
# W = np.array(
#     [[0, 0, 2, 20],
#     [0, 0, 13, 16],
#     [8, 18, 0, 0],
#     [12, 10, 0, 0]]
# )

"""
Run the discrete replicator equations for 'iter' timesteps. 

Output a timeseries of the state of the system along with a timeseries of the mean fitness.

"""
def run_sim(v0:list, iterations:int, n:int, fitness_matrix:np.array, proportionalnoise=0, constantnoise=0, format="population", training=False):
    v = v0 #BigFloat.(v0)  # The current system state
    data =  [[] for x in range(n)]  # Timeseries of system states
    meanfitness = [0]
    # Neural network training data
    # Keys = one time step, values are second timestep
    training_data = {}
    previous_reformatted_dot = None
	
    for i in range(iterations):
        # print(f"{i}/{iterations}, {str(v)}", end="\r")
        # Calculate next gen vector
        π1 = np.ones(n)
        dot = np.outer(np.array([v]), np.array([v]))
        new_v = [sum(x) for x in np.multiply(fitness_matrix, dot)]
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
        # print(f"{i}/{iterations}, {str(reformatted_dot)}, {str(v)}")
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

def run_smeared_sim(v0:list, iterations:int, n:int, proportionalnoise=0, constantnoise=0, index=None):
    # smear = [[ 0.833,  0.323, -0.157, 0.005],
    #    [-0.017 ,  0.986, 0.001,  0.029],
    #    [ 0.094,  -0.02,  0.991, -0.066],
    #    [-0.001, -0.009,  0.009,  1.001]],
    smear = [[ 0.833,  -0.017, 0.094, -0.001],
       [0.323 ,  0.986, -0.02,  -0.009],
       [-0.157,  0.001,  0.991, 0.009],
       [0.005, 0.029,  -0.066,  1.001]],
    if index:
        print(f"{index} being used!")
        model = tf.keras.models.load_model(f'./{index}/invasion_model/')
    else:
        model = tf.keras.models.load_model('./saved_models/cycling_invasion_model/')
    data = [[] for x in range(n)]
    meanfitness = [0]
    v = v0
    for i in range(iterations):
        print(f"{i}/{iterations}, {str(v)}", end="\r")
        # print(i, v)
        # new_v = np.matmul(smear, v)[0].reshape(4,)
        new_v = model(np.array([v]))[0]
        meanfitness.append(sum(new_v))
        new_v /= sum(new_v)
        # print(new_v, "new v")
        for i, val in enumerate(new_v):
            data[i].append(val)
        v = new_v
    return data, meanfitness


def runplot(format, W, index=None):
    v0 = [.25, .25, .25, .25]
    timesteps = 500
    data2x2, meanfit2x2, _ = run_sim(v0, timesteps, 4, W, format=format, constantnoise=0, proportionalnoise=0)
    plt_1 = plt.figure(figsize=(8, 8))
    if format == "population":
        for i, pop in enumerate(data2x2):
            plt.plot(pop, label=population_mapping[i])
            with open(f"frequency_logs/{population_mapping[i]}_.txt", "w") as f:
                f.write(str(pop))
        plt.legend(loc="lower right")
        if index:
            plt.savefig(f"2x2simulations/{index}/population_data.png")
        else:
            plt.savefig("graphs/population_data.png")
    elif format == "invasion":
        for i, pop in enumerate(data2x2):
            plt.plot(pop, label=invasion_mapping[i])
            with open(f"frequency_logs/{invasion_mapping[i]}_.txt", "w") as f:
                f.write(str(pop))
        # if index:
        #     data, meanfitness = run_smeared_sim(v0, timesteps, 4, index=index)
        # else:
        #     data, meanfitness = run_smeared_sim(v0, timesteps, 4)
        # for i, pop in enumerate(data):
        #     plt.plot(pop, label="modeled_"+invasion_mapping[i])
            # with open(f"frequency_logs/{invasion_mapping[i]}_.txt", "w") as f:
                # f.write(str(pop))
        plt.legend(loc="center right")
        plt.ylabel("Population Proportion")
        plt.xlabel('Timesteps')
        if index:
            plt.savefig(f"2x2simulations/{index}/invasion_data.png", bbox_inches='tight')
        else:
            plt.savefig("graphs/cycling_invasion_data.png")

def generate_training_data(format, n=500, iterations=500, fitness_matrix=None, save=False):
    # Training data generation
    # n = 1000
    # timesteps = 100
    total_training_data = {}
    for i in range(n):
        init_conditions = np.random.rand(4)
        if fitness_matrix:
            data2x2, meanfit2x2, training_data = run_sim(init_conditions, iterations, 4, fitness_matrix=fitness_matrix, constantnoise=0, proportionalnoise=0, training=format)
        else:
            data2x2, meanfit2x2, training_data = run_sim(init_conditions, iterations, 4, fitness_matrix=W, constantnoise=0, proportionalnoise=0, training=format)
        total_training_data = {**total_training_data, **training_data}
        print(f"{i+1}/{n} complete {len(total_training_data)}", end="\r")
    if save:
        with open(f"training_data/{iterations}_{format}_training_data.json", "w") as f:
            json.dump(total_training_data, f, indent=4) 
        return total_training_data
    else:
        return total_training_data

if __name__ == "__main__":
    # runplot("invasion", W, index=110)
    # runplot("invasion", W)
    for index in range(76, 176):
        with open(f"2x2simulations/{index}/W.txt", "r") as f:
            W = np.array(eval(f.read()))
        runplot("invasion", W, index=index)
    # runplot("population", W)
    # generate_training_data(W, "invasion", save=True)
    # run_smeared_sim(np.array([[0.25, 0.25, 0.25, 0.25]]).reshape(4, 1), 200, 4)