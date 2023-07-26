import sys
import re

import numpy as np
import graphviz

labels_2x2 = ["Aa", "Ab", "Ba", "Bb"]
labels_2x2x2 = ["Aaa", "Aab", "Abb", "Baa", "Bab", "Bbb"]

def generate_invasion_network(matrix, transposed=False, index=None):
    # lines = ["digraph graph {"]
    name = "invasion-network"
    if index:
        name = str(index)+"-" + name
    if len(matrix) == 6:
        name = name + "-2x2x2"
        labels = labels_2x2x2
    else:
        labels = labels_2x2
    dot = graphviz.Digraph(name, 
        comment='Invasion Network', 
        format="png",
        # engine="sfdp",
        edge_attr={'weight': '1',
                     'fontsize':'60',
                    #  'fontcolor':'blue',
                     'fontname': 'Helvetica'
                    #  'fontname': 'Lato',
                    #  'fontname': 'Times',
        },
        graph_attr={'fixedsize':'false',
                    # 'fontname': 'Times', 
                     'fontname': 'Helvetica',
                    'fontpath':'mnt/nts',
                    'concentrate':'false',
                    'beautify':'false'},
        node_attr={'fontsize':'60', 
                     'fontcolor':'black',
                     'fontname': 'Helvetica',
                    # 'fontname': 'Lato'
                    })
    dot.graph_attr['size'] = "7.75,7.75"
    dot.graph_attr['lheight'] = "100"
    dot.graph_attr['lwidth'] = "100"
    dot.graph_attr['nodesep'] = "2"
    dot.graph_attr['ranksep'] = "2"
    dot.graph_attr['dpi'] = "400"
    # dot.graph_attr['ratio'] = "fill"
    dot.graph_attr['ratio'] = "1"
    dot.graph_attr['pad'] = "0.3"
    # dot.graph_attr['mindist'] = "10"
    # dot.graph_attr['minlen'] = "10"
    # dot.graph_attr['splines'] = "curved"
    # dot.graph_attr['k'] = "10"
    # dot.graph_attr['concentrate'] = 'true'
    dot.graph_attr['overlap'] = "false"
    # dot.attr(size='5000,5000')
    # position_labels = np.array(labels).reshape(2, 2)
    # for i in range(len(position_labels)):
    #     for j in range(len(position_labels)):
    #         label = position_labels[i, j]
    #         print(i, j, label)
    #         dot.node(label, pos=str(i)+","+str(j), shape="circle") 
    for label in labels:
            dot.node(label, shape="circle") 
    max_outs = [0 for x in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # Holds the largest coefficient leaving each node
            if i == j:
                continue
            if transposed:
                max_outs[i] = max(matrix[i][j], max_outs[i])
            else:
                max_outs[j] = max(matrix[i][j], max_outs[j])
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                continue
            if transposed:
                label_i = labels[i]
                label_j = labels[j]
            else:
                label_i = labels[j]
                label_j = labels[i]
            # lines.append(label_i + " -> " + label_j)
            # size = str(min(abs(matrix[i][j])*50, 30))
            size = str(abs(matrix[i][j])*20+5)
            arrowsize = str(abs(matrix[i][j])*0.3+2)
            # arrowsize = 
            # arrowsize = str(0.3/(abs(matrix[i][j])))
            # arrowsize = size*0.3   
            # if matrix[i][j] > 0.01:
            # if matrix[i][j] > 0.005:
            # if matrix[i][j] > 0.002:
            # if matrix[i][j]-matrix[j][i] > 0.01:
            # if matrix[i][j] > 0:
            if transposed:
                if matrix[i][j] > 0.001 and matrix[i][j] > max_outs[i]/10:
                # if matrix[i][j] > 0.001:
                # if matrix[i][j] > max_outs[i]/5:
                    dot.edge(label_i, 
                            label_j, 
                            color="green3",
                            # label=str(matrix[i][j])+"-"+str((-matrix[j][i])),
                            label=str(round(matrix[i][j], 3)),
                            # label=str(round(matrix[i][j]-matrix[j][i], 3)),
                            arrowsize=arrowsize,
                            penwidth=size,
                            arrowhead='open',
                    )
                # elif matrix[i][j] < -0.01:
                #     dot.edge(label_i,
                #             label_j,
                #             color="firebrick1",
                #             label=str(round(matrix[i][j], 3)),
                #             arrowsize=arrowsize,
                #             penwidth=size
                #     )
                # else:
                    # dot.edge(label_i, label_j)
            else:
                if matrix[i][j] > 0.001 and matrix[i][j] > max_outs[j]/10:
                # if matrix[i][j] > max_outs[j]/5:
                # if matrix[i][j] > 0.001:
                    dot.edge(label_i, 
                            label_j, 
                            color="green3",
                            # label=str(matrix[i][j])+"-"+str((-matrix[j][i])),
                            label=str(round(matrix[i][j], 3)),
                            # label=str(round(matrix[i][j]-matrix[j][i], 3)),
                            arrowsize=arrowsize,
                            penwidth=size,
                            arrowhead='vee',
                    )
                # if matrix[i][j] < 0.001 and matrix[i][j] < -max_outs[j]/5:
                # # elif matrix[i][j] < -0.01:
                #     dot.edge(label_i,
                #             label_j,
                #             color="firebrick1",
                #             label=str(round(matrix[i][j], 3)),
                #             arrowsize=arrowsize,
                #             penwidth=size
                #     )
                # else:
                    # dot.edge(label_i, label_j)
    # lines.append("}")
    # s = graphviz.Source("\n".join(lines), filename="test.gv", format="png")
    # s.view()
    if index:
        if len(matrix) == 6:
            dot.render(directory=f"2x2x2simulations/{index}")
        else:
            dot.render(directory=f"2x2simulations/{index}")
    dot.render(directory="graphviz-visualizations")

def eval_matrix(string):
    regex = r"([0-9|\.|\-]+[~\s]|\]\s{1})"

    def add_comma(matchobj):
        # print(matchobj.group(all))
        return matchobj[0][:-1]+","
    
    result = re.sub(regex, add_comma, string)
    # for i in match:
        # print(i)
    matrix = np.array(eval(result)[0])
    return matrix
     

def calculate_eig(matrix, index, system="2x2", transposed=False):
    if transposed:
        new_matrix = matrix.T
        eigenvalues, eigenvectors = np.linalg.eig(new_matrix)
    else:
        new_matrix = matrix
    eigenvalues, eigenvectors = np.linalg.eig(new_matrix)
    if index:
        with open(f"{system}simulations/{index}/new_eig.txt", "w") as f:
            f.write("Eigenvalues: \n")
            f.writelines([str(np.around(x, 5))+"\n" for x in eigenvalues.tolist()])  
            f.write("\n")
            f.write("Eigenvectors: \n")
            f.writelines([str(np.around(x, 5))+"\n" for x in eigenvectors.tolist()])
            f.write("\n")
            f.write("Invasion matrix: \n") 
            f.writelines([str([float(y) for y in x])+"\n" for x in new_matrix])

if __name__ == "__main__":
    # index = 1
    if len(sys.argv) > 2:
        index = sys.argv[2]
        if sys.argv[1] == "2x2":
            # for index in range(76, 175):
            #     if index == 75:
            #         continue
            #     with open(f"2x2simulations/{index}/eig.txt", "r") as f:
            #         data = f.read()
            #     x = "".join(data.split("\n")[5:]) 
            #     matrix = eval_matrix(x)
            #     generate_invasion_network(matrix, index=index, transposed=True)
            #     calculate_eig(matrix, index)
            with open(f"2x2simulations/{index}/eig.txt", "r") as f:
                data = f.read()
            x = "".join(data.split("\n")[5:]) 
            matrix = eval_matrix(x)
            generate_invasion_network(matrix, index=index, transposed=True)
            calculate_eig(matrix, index, transposed=True)
        if sys.argv[1] == "2x2x2":
            for index in range(214, 249):
                if index == 15:
                    continue
                print(index)
                with open(f"2x2x2simulations/{index}/eig.txt", "r") as f:
                    data = f.read()
                x = np.array(eval("["+",".join(data[data.find("Invasion matrix:"):].split("\n")[1:])+"]")) 
                generate_invasion_network(x, index=index, transposed=False)
                calculate_eig(x, index, system="2x2x2")
            # if index == 15:
                # continue
            # print(index)
            # with open(f"2x2x2simulations/{index}/eig.txt", "r") as f:
            #     data = f.read()
            # x = np.array(eval("["+",".join(data[data.find("Invasion matrix:"):].split("\n")[1:])+"]")) 
            # generate_invasion_network(x, index=index, transposed=False)
            # calculate_eig(x, index, system="2x2x2")
    else:
        # matrix = [[ 0.969, -0.018,  0.053, -0.004],
        #         [ 0.114,  0.915,  0.054, -0.082],
        #         [-0.004, -0.,     1.001,  0.004],
        #         [-0.004,  0.007, -0.015,  1.012]]
        with open("matrix_storage.txt", "r") as f:
            data = f.read()
        x = "".join(data.split("\n")) 
        matrix = eval_matrix(x)
        generate_invasion_network(matrix, transposed=False)
    # generate_invasion_network(matrix, labels)