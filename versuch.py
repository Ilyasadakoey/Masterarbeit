import numpy as np
import time
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import sys

df = pd.read_excel("Mappe1.xlsx")

results = df['EtaS']


#print(results)



N_sample = 2 ** 9
cut_off = 0.005

input_data = {'num_vars': 5, 'names': ['dT', 'p_ve', 'p_e','a','b'],
              'bounds': [[2, 25], [2, 8], [200, 600],[0.5,0.7],[0.05,0.29]], 'dists': ['unif', 'unif', 'unif','unif','unif']}


sample = saltelli.sample(input_data, N = N_sample, calc_second_order=False)


np.set_printoptions(threshold=sys.maxsize)

print(sample)

#results_SA = sobol.analyze(problem=sample, Y=np.array(results), calc_second_order=False)


#print(results_SA)

#sample = np.load("ParameterArray.npy")

#np.set_printoptions(threshold=sys.maxsize)

#sampletxt = str(sample)

#with open ('samples.txt','a') as f:
    #f.write(sampletxt)


#print(sampletxt)





