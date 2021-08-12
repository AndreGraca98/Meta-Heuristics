import os, tqdm, time, math, datetime, argparse
import pandas as pd
import numpy as np
from itertools import permutations
from numba import jit, generated_jit
from numba.experimental import jitclass 
from numba import types
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# os.chdir('cio')

def load(instance=30):
    opts = {
            30: f'CIO_TSP_30_2021.txt',
            128: f'CIO_TSP_128_2021.txt',
            312: f'CIO_TSP_312_2021_correct.txt',
            }

    dataframe = pd.read_csv(opts[instance], skiprows=3, delim_whitespace=True, header=None)
    # print(dataframe.columns)
    # print(dataframe)
    # print(dataframe[0][0])

    return dataframe

@jit
def compute_cost(df, vec, start_city):
    """ compute current cost    """
    cost = 0
 
    prev_city = start_city
    for next_city in vec:
        cost += df[prev_city][next_city]
        prev_city = next_city

    cost += df[prev_city][start_city]

    return cost

@jit
def compute_cost_full_vec(df, vec):
    """ compute current cost    """
    
    cost = 0
    prev_city = vec[0]
    for city in vec[1:]:
        cost += df[prev_city][city]
        prev_city = city

    return cost

# @jit
def rand_perm(vec):
    """
        Random permutation

        Args:
            vec (np.array): cities vector

        Returns:
            vec (np.array): random cities permutation       """
    return np.random.permutation(vec)

# @jit
def rand_pairwise_perm(vec):
    """
        Random pairwise permutation

        Args:
            vec (np.array)): cities vector

        Returns:
            vec (np.array): pairwise cities permutation        """
    cityA = np.random.randint(0, len(vec))
    cityB = np.random.randint(0, len(vec))
    return swap_vec(vec, cityA, cityB)

@jit
def swap_vec(vec, a, b):
    """
        Swap 2 values in a vector

        Args:
            vec (np.array)): cities vector
            a (int): index a
            b (int): index b

        Returns:
            vec (np.array): cities vector with swaped indexes   """
    aux = vec.copy()
    aux[a], aux[b] = vec[b], vec[a]
    return aux

@jit
def slide_vec(vec, split=1):
    aux = vec[split:].copy()
    return np.append(aux, vec[:split])
    
@jit
def reverse_vec(vec):
    return vec[::-1]

def batch(iterable, n=30):
    l = math.factorial(n)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def brute_force(df, V):
    """ 
        Brute Force approach to the Traveling Salesman Problem

        Args:
            df (np.ndarray): Distances between each city
            V (np.array): Cities vector without starting city
            start_city (int): Starting city

        Returns:
            tuple(np.array, int): a tuple with the best path and cost       """

    start_city = 0

    min_cost = pow(2, 62)  # Max possible to store value
    best_cost = min_cost
    best_path = V


    l = math.factorial(df.shape[0])

    bar = tqdm.tqdm(total=l) #Initialize iterator with every permutation possible
    # bar = tqdm.tqdm(permutations(V)) #Initialize iterator with every permutation possible

    TOT_PERM = permutations(V)
    for permutation in TOT_PERM: 
    # for permutation in bar: 

        # Compute cost
        cost = compute_cost(df ,permutation, start_city)

        # Update minimum cost
        min_cost = min(min_cost, cost)

        bar.update(1)
        if min_cost < best_cost:
            best_path = permutation
            best_cost = min_cost
            bar.set_description_str(f'Minimun Cost: {min_cost} :')
            # next_permutation.write(f'Best path: {best_path}')

        
    
    return best_path, min_cost   
    
def genetic_approach(df, V):
    """
        Genetic Algorithm approach to the Traveling Salesman Problem

        Args:
            df (np.ndarray): Distances between each city
            V (np.array): Cities vector without starting city
            start_city (int): Starting city

        Returns:
            tuple(np.array, int): a tuple with the best path and cost       """
    
    min_cost = pow(2, 62)  # Max possible to store value
    best_cost = min_cost
    best_path = V

    best_individual_path = [rand_perm(V)]


    graph = np.zeros((1, NUM_IT))
    


    bar = tqdm.tqdm(range(NUM_IT))
    for it in bar:
        # Create population
        population = [rand_perm(V) for _ in range(POPSIZE)]
        population.extend(best_individual_path)
        
        best_pop_costs = []
        best_mutated_pop = []
        for p in population:
            # Mutate
            vec = [ p, 
                    reverse_vec(p), 
                    slide_vec(p, split=0), 
                    rand_pairwise_perm(p)]

            # Calculate mutation costs
            costs = []
            for v in vec:
                costs.append(compute_cost_full_vec(df, v))

            i = costs.index(min(costs))
            best_mutated_pop.append(vec[i])
            best_pop_costs.append(costs[i])

        # Select best individual from population
        i = best_pop_costs.index(min(best_pop_costs))
        best_individual_path = [best_mutated_pop[i]]
        min_individual_cost = best_pop_costs[i]

        min_cost = min(min_cost, min_individual_cost)
        if min_cost < best_cost:
            best_path = best_individual_path[0]
            best_cost = min_cost
            # bar.set_description_str(f'Minimun Cost: {best_cost} :')
            # bar.write(f'Best path: {best_path}')
        bar.set_description_str(f'Minimun Cost: {best_cost} :')

        graph[0, it] = best_cost


    return best_path, min_cost, graph


def TSP(df, algorithm='ga'):
    """
        Traveling Salesman Problem

        Args:
            df (np.ndarray): Distances between each city
            start_city (int): Starting city
            algorithm (str, optional): Algorithm to use for TSP. Defaults to 'ga'.

        Raises:
            NameError: [description]

        Returns:
            tuple(np.array, int): a tuple with the best path and cost       """
    # All cities except starting city
    V = list(range(df[0].size))
    # V = np.delete(V, start_city)    

    if algorithm=='bf':
        best_path, min_cost, graph = brute_force(df, V)
    elif algorithm=='ga':
        best_path, min_cost, graph = genetic_approach(df, V)
    else:
        raise NameError('Algorithm not implemented yet')   


    return best_path, min_cost, graph





if __name__ == '__main__':
    np.random.seed(seed=[2147483648,  875539114,  811458941, 3912185271, 3952656176 ,3015397822,
                         2382121100, 3949299757, 3705301026, 3135747426, 1422360839, 1854837366,
                         1339908117,  407517007,  375064907, 3249047552,  268302821, 1244353198,
                         1267546605, 3168979770,  867664607, 1333009265, 1802899142, 1985821909,
                          546295095, 3792563080, 3643574060, 1851324000, 2535519144, 1787358093,
                         4032882736, 4284786572, 2542287184, 3892614273, 2212388222, 2377414316,
                          419445032, 1840372660, 3131407445, 1015125321,  977955890, 1396481721,
                         1879163713, 2257837547,  551328148,  391058506, 3561071825,  521307962,
                          263311401, 3959773824, 3522030093, 2269348123, 3061650069, 3370229988,
                         2198388792, 1169880242, 3933094742,  833746484,  303580568, 1603553580,
                         4162773014, 2926929264, 1386913658, 3367917526, 3974573247, 1742878567,
                         3414953900, 1902975221, 2348916137,  522542880, 3515520506, 2966261893,
                         2148069872, 4266491848, 4148346754, 4275468078, 3016650502, 1319404449,
                         3519059920, 3045108520, 2311111384, 2118989000, 3787931815, 2453673624,
                         3704971965, 3333272744, 2466363313,  121094335, 3568254487,  977617344,
                          126397064, 4008678359, 3856507989,  271585077, 1877693627, 4203450211,
                         1399969961,  138547911, 2469617299,  152854788, 4126855378, 2603044642,
                         4059756554, 2574477243, 1511967078, 2280303727, 2179254180, 2608919544,
                         2799057054, 1963332611, 1970939626, 2579319957, 2751127394, 3932307175,
                         3712267172, 3932191032, 3550671434, 3883999620, 2126181514, 3035044591,
                         2687972879, 3034878305, 1412890294, 2401445019,  223545435,  359049315,
                         3120699090,  628881683, 1045668619, 2055312388, 2106189481, 3612836303,
                         1630977950, 3905149638, 2470057034, 1047919421, 2051368125, 3747348697,
                         3406566604, 2226451020, 3908702514,  692737615, 1865860646, 2592923940,
                         1910745595, 2509045001, 3471088260, 3964591510, 2114504032, 1124624141,
                         3145866290, 3049673038, 2968741707,  744093148, 1133552528,   82079452,
                          324826580, 1783085890,  307302401, 3661664424,  855632423, 3802672685,
                          918779491, 3337685993, 3879077150, 3569636033, 1105381650, 2092607346,
                         3734722275, 3064736553, 4259555491, 3032588729,  843970821, 2838362664,
                          594008567, 2589206652, 1234419723, 3749304828,  835824689, 3450015227,
                         2327782487,  797284224, 3246730421, 1264205906,   83072971, 1485885205,
                         2215116167, 4108422233, 1065835919, 3979650609, 4214215935,  395933425,
                         1204123037, 4266417865, 2261125350, 4087281670,  950149898, 4230147111,
                          329383377, 1825017487, 3882302623, 2497430878,  246414447, 2046196238,
                         1401949528, 4166378832, 3639775458,  296642822, 3575783482, 3141935555,
                          710222133,   95517545, 3081156558,  942451241, 1317109809, 1964915862,
                         3831228643,  625110727, 1394369564,  519350732, 1434420983, 2616468721,
                         3513316408,  275347329, 2840992910, 4142832747, 2159556376, 1195080995,
                          815308720, 1111680583, 2213160986, 2422704303, 2908854297, 2158066468,
                          849836775, 3062979817,  578996938, 1319075738, 1109303182, 2226760593,
                         2265962903, 2181473923, 1872158582, 2791713598, 3540711650,  310744192,
                         2891972688, 4038381756,  646391487, 3662294991, 3641480588, 4222426645,
                         3003528206, 3949925994,  182623282, 3492556356, 2532338088,  351545969,
                         2200075778, 2746694447, 4031893753, 3723550709, 1287257545, 3939595673,
                         1265408916, 4101135351,  694823082, 2029236235, 3455043111, 1899314736,
                         1562310158, 3506177364, 1087161044,  843942600, 3960820341, 1612817401,
                         1903852562, 2946591652, 2260293385, 2932303951,  242517964,  232694875,
                         3262564470, 4265646712, 2134724533, 2086571355, 4278745944, 2523099787,
                          187417075, 3668821544, 1000747270, 3021468431, 3617242034, 2503033652,
                         4154082666, 3524814800, 3550290405,  458520273, 4283342862, 2707248677,
                         1908958336, 4134983027, 3551932198, 2269487981, 2523098619, 1414267593,
                         2225038207,  644981801,  508751546, 3955020729, 4286369868, 1646780207,
                         2697008633, 1810304098, 2828315412,  584945765, 4016610144, 2250054936,
                         1601529509,  991871094, 1706375661, 1311686315,  180222621, 4087379512,
                         3918726203,  210857108, 3826865257, 2917945764, 1484662150, 2137087716,
                         3902078036, 4029851082, 1121479184, 2013762818, 3242816345, 2167839646,
                          469421282, 2591518962, 3124975395, 3267564059,  340401698,  774032423,
                         4033470681,  816223022, 1481682664, 3282032666, 1247253335, 1422978536,
                         4121258029,  495882891,   12036071, 1200572007, 3006244087, 3660148908,
                         3060781007, 1049791114, 2607409729, 3061269106, 1703055474, 1307678521,
                         3123324242, 2717477500,  561894205, 1220930749, 1400744245, 4279074312,
                          807866295, 4141651956, 3276000219, 2922089134, 2576417649,  456176744,
                          296119374,  855598330,  752507626, 2797608136, 4286205250, 3943771796,
                         3012486414, 2504909557, 2132389393, 1285259678, 2808201137, 1286919388,
                         4276606911,  427042801, 1000266942, 3482852608, 2816249204, 1327439130,
                         2242901727,  533319297, 4234069670,  701519445,  787457327,   25772913,
                          237591521, 1085661104, 1288985028, 3802805506,   63141851, 1410804882,
                         3793969731, 3278225085, 1341642911, 1452655248, 4118669005,  716638783,
                         3629165544, 2376856622,  898996629, 4060148792,  308924641, 1958622651,
                         2992061447, 3713194938, 1474470714, 3730637528, 3564746969, 2738244142,
                          416818928, 1214049881, 1250901631, 1710060069,  779565583, 2508498945,
                         1920496440, 3379066896, 3117690524, 3067397585, 3554555214, 3042937719,
                         3790259314, 1596000676, 1383832475,  507339365, 4128050405,  505976365,
                         4022015627, 1381778772, 1394527270, 1501982211,  690261156, 2409614119,
                         4073278657, 1460747284, 1338504215, 3619895979, 2328114332, 2088144115,
                         3469702383, 1613641654, 2202137833, 2646416824, 2970372988, 3857648551,
                         2310368845,  212794667, 3796958352, 3101587238, 2272935346,  483668433,
                          959679779, 2024623109, 2770510906, 2046579577,  394338529, 3339455956,
                         2384253556, 1696519963, 1371769715, 3969497361, 3795307151, 4250605278,
                         2041748146,  904365791, 2316768671, 2448772848, 3112970460, 2300201714,
                         1340870196, 2268429744,  477298642,  667013593,  540465950, 2503729470,
                         1777720837, 2833013780,   36229968, 3566026504, 3508847972, 4188753875,
                         4026938446, 2100288554, 2544066099, 3736036743, 3125821640,  363749675,
                         1041089261, 2547596929, 3950767251, 2482189814, 3497289796,  839962136,
                         1552269501, 1625826669, 3862699708,  134198855, 3758719745, 2684191480,
                         4179753969, 2792565787, 3873897918, 3842403203, 4292368120,  504101523,
                         2622447847, 1447765070, 2648965268, 1794818482, 3035523094, 2599028079,
                         2138428085,  894980512, 1176539624, 1694988802, 2164037612, 3548111571,
                         2280430068,   58248189, 1922996478,  315960589, 1686715873,  328988574,
                         2273524275,  639038622, 3260635105, 2989968306, 3678399807, 2385612297,
                          528356342, 3450395821, 1606463376, 4285800348, 4276627504, 3372410247,
                         1635530716, 2250564924, 1283982496, 3708140288, 1589231689,  482487562,
                         3918174194,  383273821, 2031907420,  780725744, 3542561391, 2760259459,
                           31844725,  872002513, 1000160261, 2501129105, 1981965762,  349100694,
                         4035689571, 1092229407,  623001106,  294185088, 2436317199, 2237936353,
                          464835636,   24346571,  179227302, 2837582375,  433568775, 4227103900,
                          554486100, 2841766372,  539900942, 1099468816,  129842709, 2515813873,
                          327323634, 2860884685,  285968627, 1924227892,  110282322, 4147029718,
                         1799024626, 3353131737,  200881641, 4178468774, 2494031807, 1871516936,
                         2284327240, 2089128029,  100917355,  798305680, 1841205518, 1773585982,
                         3978231718, 1743785077, 3742194189, 1367135300, 2870919830, 1947196170,
                         1897685580, 1681406416, 3615057057, 4207587407, 1749134817, 3073279798,
                         2285782294, 3482323861, 3807806214,  918343796,  686429074, 2267888808,
                         2848339846, 2894388476, 2337361177, 1493019128,  417911672, 1978673368,])
    
    parser = argparse.ArgumentParser('Ex2')
    parser.add_argument('-n', '--nodes', type=int, choices={30, 128, 312}, default=30, help='Number of nodes')
    parser.add_argument('-a', '--algorithm', type=str, choices={'ga', 'bf'}, default='ga', help='Algorithm')
    parser.add_argument('-i', '--iterations', type=float, default=4e3, help='Iterations')
    args = parser.parse_args()

    nodes = args.nodes
    alg = args.algorithm
    iterations = args.iterations

    NUM_IT = int(iterations)
    # NUM_IT = int(4e3)
    POPSIZE = 100

    dataframe = load(nodes).to_numpy()
    start_city = 0


    # alg = 'ga'  # Genetic algorithm -> ga ; Brute Force -> bf 
    best_path, min_cost, graph = TSP(df=dataframe, algorithm=alg)



    #Save data
    if not os.path.isdir('data'):
        os.makedirs('data')

    filename = f'data/TSP_{alg}_{nodes}_{iterations}'
    # filename = f'data/TSP_{datetime.datetime.now().strftime("%d%b_%H_%M")}_{alg}{nodes}'

    plt.plot(graph[0] , linewidth=1)
    plt.savefig(filename + '.png')
    plt.show()

    with open(filename+'.txt','w') as f:
        variables = {
                    'Iterations':NUM_IT,
                    'Population Size':POPSIZE,
                    'Number of Nodes':nodes,
                    }
        variables = '\n'.join([f'{k}: {v}' for k, v in variables.items() ])
        pth = ', '.join([str(n) for n in best_path])
        f.write(f"{filename[8:]}\nEx1.{alg}\n\nBest Cost: {min_cost}\nBest Path: [{pth}]\n\n\n{variables}")




    # V = list(range(dataframe[0].size))
    # V = np.delete(V, 0)   
    # print('V: ', V)
    # print('Sw:',swap_vec(V, 3, 2))
    # print('S: ',slide_vec(V))
    # print('R: ',reverse_vec(V))


    # plt.imshow(dataframe)
    # plt.show()