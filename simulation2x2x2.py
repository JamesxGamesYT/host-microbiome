import numpy as np
import json
import math
import matplotlib.pyplot as plt
import tensorflow as tf 

population_mapping = {
    0 : "A",
    1 : "B",
    2 : "a",
    3 : "b"
}
invasion_mapping = {
    0 : "Aaa",
    1 : "Aab",
    2 : "Abb",
    3 : "Baa",
    4 : "Bab",
    5 : "Bbb",
}

plt.rcParams.update({'font.size': 32})
plt.rcParams["font.family"] = "Times New Roman"


fitness_array_1 = [
   0.550406,
    0.903905,
    0.196514,
    0.0816647,
    0.29931,
    0.182285,
    0.9562,
    0.26528,
    0.0560907,
    0.799666,
    0.682008,
    0.533225,
    0.410043,
    0.293506,
]

fitness_array_2 = [
   0.873075,
    0.674558,
    0.292395,
    0.399689,
    0.664959,
    0.43416,
    0.42416,
    0.581351,
    0.117506,
    0.453779,
    0.905191,
    0.4042,
    0.624763,
    0.418543,
]

fitness_array_3 = [0.941211,0.937435,
    0.114152, 0.338324, 0.318469,
    0.407763, 0.558483, 0.50553,
    0.397596, 0.712525, 0.334249,
    0.497081, 0.281798, 0.417097,
]

fitness_array_4 = [0.93481,0.940011,
    0.289972, 0.496797, 0.03345111,
    0.719061, 0.59928, 0.894289,
    0.191121, 0.454242, 0.694832,
    0.194553, 0.0499185, 0.229476,
]

fitness_array_5 = [0.775892,0.949422,
    0.0612043, 0.11011, 0.07140439,
    0.192017, 0.429641, 0.41338,
    0.658555, 0.53601, 0.734546,
    0.0766426, 0.0194165, 0.090286,
]

fitness_array_6 = [0.776693,0.778034,
    0.537126, 0.572795, 0.966995,
    0.151988, 0.279967, 0.72284,
    0.446017, 0.650643, 0.726204,
    0.802337, 0.974784, 0.186716,
]

fitness_array_7 = [0.54082,0.185669,
    0.496694, 0.277161, 0.270272,
    0.169313, 0.984411, 0.255051,
    0.174892, 0.645024, 0.846532,
    0.789934, 0.285838, 0.63831,
]

fitness_array_8 = [0.0412494,0.130777,
    0.975035, 0.829493, 0.979969,
    0.831812, 0.128972, 0.950661,
    0.658599, 0.0341865, 0.343056,
    0.859581, 0.655942, 0.253042,
]

fitness_array_9 = [0.256637,0.605387,
    0.614418, 0.426934, 0.585161,
    0.0451742, 0.132471, 0.561748,
    0.229797, 0.785912, 0.619411,
    0.316413, 0.654484, 0.0527539,
]

fitness_array_10 = [0.377255,0.786649,
    0.495698, 0.179468, 0.496816,
    0.840895, 0.165409, 0.926327,
    0.636382, 0.0192653, 0.189524,
    0.49246, 0.0603908, 0.994589,
]

fitness_array_11 = [0.365744,0.973911,
    0.31667, 0.847, 0.290089,
    0.502121, 0.821505, 0.139997,
    0.13768, 0.945691, 0.682255,
    0.833348, 0.256225, 0.290185,
]

fitness_array_12 = [0.288754,0.256865,
    0.480592, 0.924153, 0.161479,
    0.538621, 0.580268, 0.69925,
    0.00940748, 0.209365, 0.0767278,
    0.343853, 0.882398, 0.836969,
]

fitness_array_13 = [0.461575,0.791982,
    0.886053, 0.62728, 0.123756,
    0.113301, 0.988199, 0.708863,
    0.268214, 0.40054, 0.362525,
    0.0876326, 0.216698, 0.953333,
]

fitness_array_14 = [0.169111,0.182331,
    0.885583, 0.748, 0.440206,
    0.0496607, 0.156076, 0.965304,
    0.476305, 0.596936, 0.51224,
    0.218504, 0.361693, 0.824279,
]

fitness_array_15 = [0.997152,0.155826,
    0.98835, 0.829048, 0.443127,
    0.984935, 0.670391, 0.347852,
    0.256821, 0.866822, 0.145378,
    0.0328725, 0.660866, 0.429852,
]

fitness_array_16 = [0.356927,0.0814709,
    0.979065, 0.902662, 0.12722,
    0.98787, 0.503801, 0.940182,
    0.303392, 0.988931, 0.0120134,
    0.90154, 0.0404422, 0.488651,
]

fitness_array_17 = [0.41141,0.866074,
    0.533471, 0.807085, 0.385072,
    0.0361167, 0.428608, 0.713338,
    0.66209, 0.036419, 0.334615,
    0.333605, 0.372529, 0.395571,
]

fitness_array_18 = [0.158126,0.380078,
    0.770818, 0.631565, 0.967234,
    0.932228, 0.728752, 0.548445,
    0.764281, 0.773263, 0.374127,
    0.888963, 0.780426, 0.746155,
]

fitness_array_19 = [0.929127,0.472,
    0.14371, 0.212574, 0.476859,
    0.859857, 0.339775, 0.646551,
    0.0982689, 0.16687, 0.22999,
    0.810645, 0.311247, 0.365997,
]

fitness_array_20 = [0.606326,0.660055,
    0.740125, 0.71643, 0.437038,
    0.952289, 0.349793, 0.878276,
    0.708146, 0.0786811, 0.521581,
    0.0640205, 0.669989, 0.574037,
]

fitness_array_21 = [0.400417,0.371448,
    0.89042, 0.683794, 0.89116,
    0.234793, 0.577901, 0.829181,
    0.972467, 0.571119, 0.549224,
    0.256699, 0.602095, 0.999633,
]

fitness_array_22 = [0.395209,0.0375313,
    0.746855, 0.25744, 0.751626,
    0.290132, 0.0831138, 0.97398,
    0.244899, 0.178536, 0.0690118,
    0.484243, 0.429886, 0.615725,
]

fitness_array_23 = [0.053109,0.323225,
    0.590185, 0.394789, 0.999054,
    0.526977, 0.152647, 0.587404,
    0.560592, 0.512971, 0.384815,
    0.0427271, 0.446354, 0.187894,
]

fitness_array_24 = [0.607759,0.674933,
    0.0784851, 0.197149, 0.247537,
    0.166283, 0.564717, 0.599004,
    0.179742, 0.00860838, 0.742164,
    0.843028, 0.446826, 0.188095,
]

fitness_array_95 = [0.3322515896920779, 0.6422464057947375, 0.26965989310015503, 0.3436687705365151, 0.003487726405320468, 0.043981722916557775, 0.7023464881506192, 0.4428583947723792, 0.9068731121002681, 0.110372693552233, 0.6216156878101181, 0.7991332007480705, 0.29935610819249237, 0.6921754150949305]

fitness_array_141 = [0.5089595349014889, 0.14824224507955464, 0.5082983823254468, 0.40868172517411483, 0.10560850352340578, 0.8442240807779349, 0.1396058275456793, 0.38356286039927423, 0.6758089836213016, 0.9055660272677218, 0.08545080211086131, 0.09362000677760274, 0.41399911149679225, 0.6570142599840546]

fitness_array_143 = [0.03786586832579664, 0.9350942681339585, 0.5264351759010372, 0.08739457860226796, 0.16548533639397645, 0.6146505823480957, 0.1555211429874922, 0.6168606232373047, 0.5036639453390057, 0.9129478945560342, 0.3485732585515925, 0.32904083005535, 0.06199820662009403, 0.5557856717849546]

fitness_array_144 = [0.7710983390389716, 0.6047519582766018, 0.05821470587048361, 0.4047066045006533, 0.6423147583392429, 0.4074827846790865, 0.5206641876407337, 0.3610757387586232, 0.2189870857371402, 0.9665390215660588, 0.7384124374920485, 0.6938454856563948, 0.19702619147339695, 0.41227813527148327]

fitness_array_146 = [0.7359225353244541, 0.035785053562307545, 0.8998251764588606, 0.5325722122867196, 0.8027189262815861, 0.19860009105264598, 0.355550984780325, 0.5308578799513612, 0.04483663785982217, 0.01093916803195294, 0.8439544357007622, 0.009081919821318674, 0.013713017362749658, 0.052494395354841794]

fitness_array_147 = [0.8467565995877597, 0.8087156111511935, 0.2850490462062638, 0.03878316921891434, 0.5369270668056502, 0.9661047745957367, 0.1851215934274678, 0.9555821096430518, 0.3420690439593661, 0.42199169613627086, 0.8385745651397427, 0.2604762488831177, 0.526845356050868, 0.21938137006153047]

fitness_array_155 = [0.40258038370493043, 0.4583364626856491, 0.21876775943915194, 0.7536309238000267, 0.0005967032909380832, 0.9421079716903983, 0.6344387617628595, 0.9813998793752269, 0.6462831817264877, 0.8144926665163627, 0.4820693708143369, 0.41564585367665163, 0.004009850235519585, 0.3759155940021167]

fitness_array_159 = [0.06584080917782387, 0.9128722972052483, 0.6074744898475068, 0.3124498524542464, 0.25790797228881845, 0.25928433918821436, 0.26900919928499023, 0.9253162099230805, 0.767504633886019, 0.2036955910701831, 0.36610504816710265, 0.4582375552955543, 0.9423479329891778, 0.5230920146362343]

fitness_array_168 = [0.5161683808502493, 0.47565618394760023, 0.7049419806221021, 0.3659344892175266, 0.6331022537061022, 0.5385943670033196, 0.2887114408013569, 0.543334726615196, 0.06950920344305267, 0.4525685357867477, 0.2632649174497571, 0.5749189039246884, 0.238998041567261, 0.12048665801633951]

fitness_array_174 = [0.9254261327161667, 0.09018043843450463, 0.44131784323359635, 0.8693078383398106, 0.9203441320198759, 0.9649501737119659, 0.707014921863185, 0.4420959559520037, 0.056445270378182366, 0.06455862843702742, 0.19116732457991203, 0.17048348119775136, 0.48069794673714195, 0.7254706331031671]

fitness_array_196 = [0.6092758075816938, 0.8771003358807583, 0.6876172378075436, 0.7628586392077289, 0.7063454928764284, 0.07080172634089654, 0.9315706626786847, 0.875690669011065, 0.8773477241310017, 0.6020993280536953, 0.4351455021475563, 0.8668378815091425, 0.5717680787646267, 0.8038354850924561]

fitness_array_209 = [0.6686280963332543, 0.06523711888390182, 0.17130371189915305, 0.7679868960671952, 0.9822932718736008, 0.9970217886912707, 0.4105166916871339, 0.5355536368074251, 0.09889464448619445, 0.4271626832722438, 0.2866933500044734, 0.5642015674743047, 0.37581336738318516, 0.11753413100406651]

def run_sim(v0, iterations, fitness_array, format="population", training=False):
    v = v0
    meanfitness = [0]

    if format == "population":
        data =  [[] for x in range(4)]  # Timeseries of system states
    else:
        data =  [[] for x in range(6)]  # Timeseries of system states
    
    previous_invasion_v = None
    training_data = {}
    waAa, waAb, waBa, waBb, wAaa, wAab, wAbb, wbAa, wbAb, wbBa, wbBb, wBaa, wBab, wBbb = fitness_array
    def AUp(A, B, a, b):
        return A*(
        0*A*A + 0*A*B + 0*B*B +
        0*A*a + 0*A*b + 0*B*a + 0*B*b +
        wAaa*a*a + 2*wAab*a*b + wAbb*b*b
        )

    def BUp(A, B, a, b):
        return B*(
        0*A*A + 0*A*B + 0*B*B +
        0*A*a + 0*A*b + 0*B*a + 0*B*b +
        wBaa*a*a + 2*wBab*a*b + wBbb*b*b
        )

    def aUp(A, B, a, b):
        return a*(
        0*A*A + 0*A*B + 0*B*B + 
        waAa*A*a + waAb*A*b + waBa*B*a + waBb*B*b +
        0*a*a + 0*a*b + 0*b*b
        )

    def bUp(A, B, a, b):
        return b*(
        0*A*A + 0*A*B + 0*B*B +
        wbAa*A*a + wbAb*A*b + wbBa*B*a + wbBb*B*b +
        0*a*a + 0*a*b + 0*b*b
        )

    def update(A, B, a, b):
        # Update the population vector and normalize
        current = np.array([AUp(A, B, a, b), BUp(A, B, a, b), aUp(A, B, a, b), bUp(A, B, a, b)])
        current = current / sum(current)
        return current
    
    for i in range(iterations):
        v = update(*v)
        # print(v)
        invasion_v = [
            v[0]*v[2]*v[2], # Aaa
            v[0]*v[2]*v[3]*2, # Aab
            v[0]*v[3]*v[3], # Abb
            v[1]*v[2]*v[2], # Baa
            v[1]*v[2]*v[3]*2, # Bab
            v[1]*v[3]*v[3]
            ]
        invasion_v /= sum(invasion_v)
        # print(invasion_v, v)
        if training == "population":
            training_data[str(v.tolist())] = tuple(v)
            meanfitness.append(sum(v))  # Record mean fitness
        elif training == "invasion":
            if i >= 1:
                diff = math.sqrt(sum((previous_invasion_v-invasion_v)**2))
                # if diff > 0.001:
                    # print(previous_invasion_v, invasion_v, diff, i)
                    # training_data[str(previous_invasion_v.tolist())] = invasion_v.tolist()
                    # meanfitness.append(sum(invasion_v))  # Record mean fitness
                # if diff > 0.001:
                # print(previous_invasion_v, invasion_v, diff, i)
                training_data[str(previous_invasion_v.tolist())] = invasion_v.tolist()
                meanfitness.append(sum(invasion_v))  # Record mean fitness

        previous_invasion_v = invasion_v
        if format == "population":
            for i in range(4):
                data[i].append(v[i]) # Add main sim data to timeseries
        elif format == "invasion":
            for i in range(6):
                data[i].append(invasion_v[i]) # Add main sim data to timeseries
    return data, meanfitness, training_data

def run_smeared_sim(v0, timesteps, index=None):
    if index:
        print(f"{index} being used!")
        model = tf.keras.models.load_model(f'./2x2x2simulations/{index}/invasion_model/')
    else:
        model = tf.keras.models.load_model('./saved_models/cycling_invasion_model/')
    data = [[] for x in range(6)]
    meanfitness = [0]
    v = v0
    for i in range(timesteps):
        print(f"{i}/{timesteps}, {str(v)}", end="\r")
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
    timesteps = 1000
    data2x2x2, meanfitness, _ = run_sim(v0, timesteps, W, format=format)
    plt_1 = plt.figure(figsize=(15, 8))
    # plt_1 = plt.figure(figsize=(8, 8))
    if format == "population":
        for i, pop in enumerate(data2x2x2):
            plt.plot(pop, label=population_mapping[i])
            with open(f"frequency_logs/{population_mapping[i]}_2x2x2.txt", "w") as f:
                f.write(str(pop))
        plt.legend(loc="lower right")
        if index:
            plt.savefig(f"2x2x2simulations/{index}/population_data.png")
        else:
            plt.savefig("graphs/population_data_2x2x2.png")
    elif format == "invasion":
        for i, pop in enumerate(data2x2x2):
            plt.plot(pop, label=invasion_mapping[i])
            with open(f"frequency_logs/{invasion_mapping[i]}_2x2x2.txt", "w") as f:
                f.write(str(pop))
        v0 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        if index:
            data, meanfitness = run_smeared_sim(v0, timesteps, index=index)
        else:
            data, meanfitness = run_smeared_sim(v0, timesteps)
        for i, pop in enumerate(data):
            plt.plot(pop, label="modeled_"+invasion_mapping[i])
            with open(f"frequency_logs/{invasion_mapping[i]}_.txt", "w") as f:
                f.write(str(pop))
        # plt.legend(loc="center right")
        plt.ylabel("Population Proportion")
        plt.xlabel('Timesteps')
        if index:
            plt.savefig(f"2x2x2simulations/{index}/invasion_data.png", bbox_inches='tight')
        else:
            plt.savefig("graphs/cycling_invasion_data_2x2x2.png", bbox_inches='tight')

def generate_training_data(format, n=500, iterations=1000, fitness_array=None, save=False):
    # Training data generation
    # n = 1000
    # timesteps = 100
    total_training_data = {}
    for i in range(n):
        init_conditions = np.random.rand(4)
        if fitness_array:
            data2x2, meanfit2x2, training_data = run_sim(init_conditions, iterations, fitness_array=fitness_array, training=format)
        else:
            data2x2, meanfit2x2, training_data = run_sim(init_conditions, iterations, fitness_array=fitness_array_196, training=format)
        total_training_data = {**total_training_data, **training_data}
        print(f"{i+1}/{n} complete {len(total_training_data)}", end="\r")
    if save:
        with open(f"training_data/{iterations}_{format}_training_data_2x2x2.json", "w") as f:
            json.dump(total_training_data, f, indent=4) 
        return total_training_data
    else:
        return total_training_data
    
if __name__ == "__main__":
    v0 = [.25, .25, .25, 25]

    # run_sim(v0, 1000, 4, fitness_array)
    # for i in range(1, 24):
        # runplot("population", eval("fitness_array_"+str(i)), index=i)
        # runplot("invasion", eval("fitness_array_"+str(i)), index=i)
    # runplot("invasion", fitness_array_10, index=10)
    # runplot("invasion", fitness_array_1, index=1)
    index = 142
    with open(f"2x2x2simulations/{index}/W.txt", "r") as f:
        W = eval(f.read())
    runplot("invasion", W, index=index)