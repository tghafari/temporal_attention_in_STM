# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:03:29 2020

@author: sakarimi
"""
# get ANOVA table as R like output
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import functions as fcn
from statsmodels.stats.multicomp import pairwise_tukeyhsd

######################### Quadrant Data ######################
# df2 = pd.read_csv('Data\PooledDataQuadrants data\PooledDataQuadrants.csv')
# data = df2[['score','reaction_time','Quad','predictability', 'difficulty','foreperiod','TP', 'TN', 'FP', 'FN','xpos','ypos']]


########################## 4diff Data ########################
df2 = pd.read_csv('Data\PooledData4DiffLev data\PooledData4DiffLev.csv')
data = df2[['score','reaction_time','predictability', 'difficulty','foreperiod','TP', 'TN', 'FP', 'FN','xpos','ypos']]
q = [1 if (i>0 and j>0) else (2 if (i<0 and j>0) else (3 if (i<0 and (j<0)) else 4))
      for (i,j) in zip(data.xpos,data.ypos)] # Gives 4 instead on NaN
data['Quad'] = q


###################### Data Cleaning #########################
data.predictability = ['a' if i==3 else 'b' for i in data.predictability] # which was which?
data.foreperiod = ['S' if i==0.65 else 'L' for i in data.foreperiod]
# topbot = [1 if (i==1 or i==2) else 0 for i in data.Quad]
data['top'] = [1 if (i==1 or i==2) else 0 for i in data.Quad]
data['right'] = [1 if (i==1 or i==4) else 0 for i in data.Quad]

# Q = ['TR','TL','BL','BR']
# data.Quad = [Q[i-1] for i in data.Quad]
data.difficulty = [int((i-4)/2) for i in data.difficulty]


######################## Box Plot #############################
ax = sns.boxplot(x='Quad', y='reaction_time', data=data, color='#99c2a2')
# ax = sns.swarmplot(x="Quad", y="reaction_time", data=data, color='#7d0013')


##################### Removing outliers #######################
data = data[data['reaction_time']<=1.5]
data = data.round(3)


########################### ANOVA #############################
data.newq = data.TP+data.FN
data1 = data[data.newq==1]

model = ols('reaction_time ~ C(Quad)', data=data1).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('reaction_time ~ C(top)', data=data1).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('reaction_time ~ C(right)', data=data1).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model = ols('reaction_time ~ C(ypos)', data=data1).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


######################## Post Hoc #############################
res = pairwise_tukeyhsd(data1['reaction_time'], data1['Quad'])
print(res.summary())

res = pairwise_tukeyhsd(data1['reaction_time'], data1['ypos'])
print(res.summary())

#################### Trends in degrees ########################
data2 = data[data.TP==1]
X = data2[['xpos','ypos']]
y = data2[['reaction_time']]
model = fcn.lin_reg2(X,y,t_size=0)
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'ypos', fig=fig)


######################### Performance ##########################
perfQ = []
for i in range(1,5):
    dat = data1[data.Quad==i]
    perfQ.append(sum(dat.score)/len(dat))
    del dat

perfT = [perfQ[0]+perfQ[1], perfQ[2]+perfQ[3]]


