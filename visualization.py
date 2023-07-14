import sys
import re

import numpy as np
import graphviz

labels = ["Aa", "Ab", "Ba", "Bb"]

def generate_invasion_network(matrix, labels, transposed=False, index=None):
    # lines = ["digraph graph {"]
    if index:
        name = str(index)+"-invasion-network"
    else:
        name = 'invasion-network'
    print(name, "name")
    dot = graphviz.Digraph(name, 
        comment='Invasion Network', 
        format="png",
        engine="fdp",
        edge_attr={'weight':'1',
                     'fontsize':'60',
                    #  'fontcolor':'blue',
                    #  'fontname': 'Helvetica'
                     'fontname': 'Lato',
        },
        graph_attr={'fixedsize':'false', 
                    },
        node_attr={'fontsize':'60', 
                     'fontcolor':'black',
                    'fontname': 'Lato'})
    dot.graph_attr['size'] = "7.75,10.25"
    dot.graph_attr['lheight'] = "100"
    dot.graph_attr['lwidth'] = "100"
    dot.graph_attr['nodesep'] = "3"
    dot.graph_attr['ranksep'] = "3"
    dot.graph_attr['dpi'] = "200"
    dot.graph_attr['ratio'] = "1"
    dot.graph_attr['pad'] = "0.3"
    # dot.attr(size='5000,5000')
    position_labels = np.array(labels).reshape(2, 2)
    print(position_labels)
    for i in range(len(position_labels)):
        for j in range(len(position_labels)):
            label = position_labels[i, j]
            print(i, j, label)
            dot.node(label, pos=str(i)+","+str(j), shape="circle") 
    # for label in labels:
            # dot.node(label) 
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
            size = str(abs(matrix[i][j])*30+5)
            arrowsize = str(abs(matrix[i][j])*1+1)
            # arrowsize = 
            # arrowsize = str(0.3/(abs(matrix[i][j])))
            # arrowsize = size*0.3   
            if matrix[i][j] > 0.01:
                dot.edge(label_i, 
                        label_j, 
                        color="green3",
                        label=str(matrix[i][j]),
                        arrowsize=arrowsize,
                        penwidth=size,
                )
            # elif matrix[i][j] < -0.01:
            #     dot.edge(label_i,
            #             label_j,
            #             color="firebrick1",
            #             label=str(matrix[i][j]),
            #             arrowsize=arrowsize,
            #             penwidth=size
            #     )
            # else:
                # dot.edge(label_i, label_j)
    # lines.append("}")
    # s = graphviz.Source("\n".join(lines), filename="test.gv", format="png")
    # s.view()
    print(dot.source)
    dot.render(directory="graphviz-visualizations")

def generate_from_index(index):
    with open(f"{index}/eig.txt", "r") as f:
        data = f.read()
    x = "".join(data.split("\n")[5:])  
    print(x)
    regex = r"([0-9|\.|\-]+[~\s]|\]\s{1})"

    def add_comma(matchobj):
        print(matchobj[0] == ']', matchobj[0], type(matchobj[0]))
        # print(matchobj.group(all))
        return matchobj[0][:-1]+","
    
    result = re.sub(regex, add_comma, x)
    # for i in match:
        # print(i)
    print(result)
    matrix = np.array(eval(result)[0])
    return matrix

if __name__ == "__main__":
    # index = 151
    if len(sys.argv) > 1:
        index = sys.argv[1]
        matrix = generate_from_index(index)
        generate_invasion_network(matrix, labels, index=index, transposed=True)
    # generate_invasion_network(matrix, labels)