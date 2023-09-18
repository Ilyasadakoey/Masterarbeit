# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:09:31 2023

@author: Max
"""

# Import Libraries
import numpy as np
import os
import math
import datetime
import sys
import pickle
import pandas as pd
# Import Saltelli-SampleSet-Function
from SALib.sample import saltelli

### Definiere Saltelli-Problem - Dictionary ###
problem = {'num_vars': 5, 'names': ['dT', 'p_ve', 'p_e','a','b'],
              'bounds': [[2, 25], [2, 8], [200, 600],[0.5,0.7],[0.05,0.29]], 'dists': ['unif', 'unif', 'unif','unif','unif']}

# Erstellt eine Variable "dir_path", die den absoluten Pfad des Verzeichnisses enthält, in dem die Python-Datei ausgeführt wird.
dir_path = os.path.dirname(os.path.realpath(__file__))

# Lade Dictionary
# problem_load = pickle.load(open(dir_path + "/problem_dict.p", "rb"))

# Number of samples (input) created for S.A.; -> N * (D + 2)
N_sample = 2 ** 9

# Totale Saltelli-Sample-Verteilung
N_total = N_sample * (problem['num_vars'] + 2)

# Erstelle eine Stichprobe von Eingangsparametern für die Analyse der globalen Sensitivität eines mathematischen Modells. Das Saltelli-Sampling ist eine Methode der quasi-Monte-Carlo-Stichprobenziehung
sample_set = saltelli.sample(problem=problem, N=N_sample, calc_second_order=False)

# Anzahl an einzelnen Arrays
parts = math.ceil(N_total / 225)



# Erstellt neues Verzeichnis und speichert die Saltelli-Stichprobe in diesem Verzeichnis
date = str((datetime.datetime.now()).strftime("%Y%m%d-%H%M%S"))
file_out = 'Splitted_Sample_Set/' + date + '/'
os.makedirs(file_out)
np.save(file_out + 'ParameterArray', sample_set)

# Speichert das Python-Objekt "problem" als Pickle-Datei im Verzeichnis "file_out".
pickle.dump(problem, open(file_out + "problem_dict.pkl", "wb"))


# Erstelle Text-Datei mit Saltelli-Problem
filename = "SaltelliProblem.txt"
file = open(file_out + filename, "w")

for line in range(problem['num_vars']):
    text = problem['names'][line] + ', bounds: ' + str(problem['bounds'][line]) + ', distribution: ' + problem['dists'][
        line] + '\n'
    file.write(text)

file.write('N_sample: ' + str(N_sample) + ', Total SampleSize: ' + str(N_total))
file.close()

# Unterteile die gesamte Stichprobe in "parts" (Anzahl) Teile
split = np.array_split(sample_set, parts)

# Erstelle und speichere einzelne Arrays aus split-Array
for i in range(len(split)):
    splitted_array = split[i]
    array_name = file_out + '/Split_' + str(i)
    np.save(array_name, splitted_array)

# Erstelle DataFrame zur Speicherung der Rechenzeiten
statistics = pd.DataFrame({'Parameteranzahl': [], 'Rechenzeit': []})
statistics.to_pickle(file_out + '/Statistik.pkl')



