import numpy as np
import matplotlib.pyplot as plt

def autoclassify(index):
    # if i in [206, 228, 236, 187, 212, 214, 218, 219]:
        # continue
    with open(f"2x2x2simulations/{index}/W.txt", "r") as f:
        W = eval(f.read())
    if W == None:
        return None
    with open(f"2x2x2simulations/{index}/eig.txt", "r") as f:
        data = f.read()
    x = np.array(eval("["+",".join(data[data.find("Invasion matrix:"):].split("\n")[1:])+"]")) 
    sink_nodes = 0
    max_columns = [0]*6
    stop = False
    for y in range(len(x)):
        sink_node = True
        for j in range(len(x)):
            if j == y:
                continue
            # print(x[j][y], j, y)
            if x[j][y] > 0.001:
                sink_node = False
            max_columns[y] = max(max_columns[y], x[j][y])
        if sink_node == True:
            sink_nodes += 1
            if y > 2:
                stop = True
        for j in range(len(x)):
            if x[j][y] < max_columns[y]/10:
                x[j][y] = 0
    if stop == True:
        return None
    # if i == 210:
    coexistence = False
    for j in range(len(x)):
        for k in range(len(x)):
            if j == k:
                continue
            if x[j][k] > 0.001 and x[k][j] > 0.001:
                coexistence = True
                # print(j, k)
    return sink_nodes, coexistence
    # if sink_nodes == 0 and coexistence:
        # plt.scatter(W[8]-W[0], W[7]-W[1], color="red")
        # plt.scatter(W[8]-W[1], W[7]-W[0], color="red")
    # elif sink_nodes == 1 and not coexistence:
        # plt.scatter(W[8]-W[0], W[7]-W[1], color="blue")
        # plt.scatter(W[8]-W[1], W[7]-W[0], color="blue")
    # elif sink_nodes == 1 and coexistence:
        # plt.scatter(W[8]-W[0], W[7]-W[1], color="green")
        # plt.scatter(W[8]-W[1], W[7]-W[0], color="green")
    # elif sink_nodes == 2 and not coexistence:
        # plt.scatter(W[8]-W[0], W[7]-W[1], color="orange")
        # plt.scatter(W[8]-W[1], W[7]-W[0], color="orange")
    # elif sink_nodes == 0 and not coexistence:
        # plt.scatter(W[8]-W[0], W[7]-W[1], color="orange")
        # plt.scatter(W[8]-W[1], W[7]-W[0], color="purple")
# plt.xlabel("wbAb - waAb")
# plt.ylabel("wbAa - waAa")
# plt.axvline(x=0)
# plt.axvline(y=0)
# plt.savefig("plot.png")