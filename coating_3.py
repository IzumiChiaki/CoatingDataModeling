# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:07:18 2019

@author: Chiaki
"""

import sys
sys.path.append("D:\Python\coating")

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minepy import MINE
get_ipython().run_line_magic('matplotlib', 'inline') #plot inline

#######compute pearson and mic
df = pd.read_csv('./coating.csv')
pearson_list = []
mic_list = []
mine = MINE(alpha=0.6, c=15, est='mic_approx')
for i in range(df.values.shape[1]-2):
    a = np.array(pearsonr(df.values[:,i+1], df.values[:,5]))
    pearson_list.append(a)
    b = mine.compute_score(df.values[:,i+1], df.values[:,5])
    mic_list.append(mine.mic())
pearson = np.array(pearson_list)
mic = np.array(mic_list)
print(pearson[:,0])
print(mic)

########plot heatmap
dfData = df.corr()
plt.subplots(figsize=(9, 9)) #set figsize
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
plt.savefig('./BluesStateRelation.png')
plt.show()

k = [[0.822, 0.245, 0.009],[0.459, 0.07, 0.003],[0.971, 0.008, 0.006]]
E = df.values[:,1:4].astype(np.float64).dot(np.array(k).T)
pr_list = []
for i in range(np.array(k).shape[0]):
    pr_list.append(pearsonr(E[:,i], df.values[:,5]))
pr = np.array(pr_list)[:,0]
#print(pr)
list_a = pr.tolist()
max_index = list_a.index(max(list_a))
print("==================Emax===================")
print(max(list_a))
print("====================k====================")
print(k[max_index])