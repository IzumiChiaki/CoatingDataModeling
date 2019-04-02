# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:17:09 2019

@author: Chiaki
"""

import sys
sys.path.append("D:\Python\coating")

from scipy.stats import pearsonr
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minepy import MINE

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
print("==================pearson_r===================")
print(pearson[:,0])
print("=====================mic======================")
print(mic)

########plot heatmap
dfData = df.corr()
plt.subplots(figsize=(9, 9)) #set figsize
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
plt.savefig('./BluesStateRelation.png')
plt.show()

k = [[0.2151, 0.1167, 0.1341, 0.437],[0.8433, 0.6807, 0.2327, 0.8146],\
     [0.4965, 0.5882, 0.2571, 0.9596],[0.8664, 0.7454, 0.2311, 0.8309],\
     [0.9246, 0.535, 0.2623, 0.8341],[0.3206, 0.6847, 0.2863, 0.9878],\
     [0.2117, 0.6793, 0.252, 0.9549],[0.0966, 0.1131, 0.2272, 0.8107],\
     [0.8951, 0.0045, 0.1323, 0.5011],[0.1044, 0.0484, 0.1355, 0.5278],\
     [0.1088, 0.3422, 0.2434, 0.8759]]
E_list = df.values[:,1:5].astype(np.float64).dot(np.array(k).T)
pr_list = []
for i in range(np.array(k).shape[0]):
    pr_list.append(pearsonr(E_list[:,i], df.values[:,5]))
pr = np.array(pr_list)[:,0]
list_a = pr.tolist()
max_index = list_a.index(max(list_a))
print("==================E_GLR_max===================")
print(max(list_a))
print("======================k=======================")
print(k[max_index])


E = E_list[:,max_index]
GLR = df.values[:,5]


def func_linear(x, p):
    u, v = p
    return u * x + v

def residuals_linear(p, y, x):
    return y - func_linear(x, p)

def func_curve(x, p):
    u, v, w = p
    return u * x * x + v * x + w

def residuals_curve(p, y, x):
    return y - func_curve(x, p)

def func_exp(x, u, v):
    return u * np.exp(v*x)


p_linear = [1, 1]
plsq_linear = leastsq(residuals_linear, p_linear, \
               args=(np.array(GLR).astype(np.float64), np.array(E).astype(np.float64)))
print("==================Linear_para===================")
print(plsq_linear[0])
print("===============Linear_expression================")
print('GLR='+str(plsq_linear[0][0])+'E'+str(plsq_linear[0][1]))

p_curve = [1, 1, 1]
plsq_curve = leastsq(residuals_curve, p_curve, \
               args=(np.array(GLR).astype(np.float64), np.array(E).astype(np.float64)))
print("==================Curve_para===================")
print(plsq_curve[0])
print("===============Curve_expression================")
print('GLR='+str(plsq_curve[0][0])+'E*E+'+str(plsq_curve[0][1])+'E'+str(plsq_curve[0][2]))

plsq_exp = curve_fit(func_exp, np.array(GLR).astype(np.float64), np.array(E).astype(np.float64))
print("==================Exp_para===================")
print(plsq_exp[0])

####plot linear E-GLR
plt.figure(figsize=(10, 5), facecolor='w')
plt.plot(E, GLR, 'ro', lw=2, markersize=6)
plt.grid(b=True, ls=':')
plt.xlabel(u'E', fontsize=16)
plt.ylabel(u'GLR', fontsize=16)
plt.plot(E, func_linear(E, plsq_linear[0]))
plt.legend(("real data","fitting data"))
plt.savefig('./E_GLR_linear.png')
plt.show()


E_fit = np.linspace(min(E),max(E),100)
####plot curve E-GLR
plt.figure(figsize=(10, 5), facecolor='w')
plt.plot(E, GLR, 'ro', lw=2, markersize=6)
plt.grid(b=True, ls=':')
plt.xlabel(u'E', fontsize=16)
plt.ylabel(u'GLR', fontsize=16)
plt.plot(E_fit, func_curve(E_fit, plsq_curve[0]))
plt.legend(("real data","fitting data"))
plt.savefig('./E_GLR_curve.png')
plt.show()

####plot exp E-GLR
plt.figure(figsize=(10, 5), facecolor='w')
plt.plot(E, GLR, 'ro', lw=2, markersize=6)
plt.grid(b=True, ls=':')
plt.xlabel(u'E', fontsize=16)
plt.ylabel(u'GLR', fontsize=16)
plt.plot(E_fit, func_exp(E_fit, 8.7252, 0.0008))
plt.legend(("real data","fitting data"))
plt.savefig('./E_GLR_exp.png')
plt.show()