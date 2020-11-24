# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:36:45 2020

@author: sakarimi
"""
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# import fcns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

os.getcwd()
os.listdir()

# df2 = pd.read_csv('Dropbox\Tara\Data\PooledData4DiffLev data\PooledData4DiffLev.csv')
df2 = pd.read_csv('Data\PooledDataQuadrants data\PooledDataQuadrants.csv')


data = df2[['score','reaction_time','Quad','predictability', 'difficulty','foreperiod','TP', 'TN', 'FP', 'FN']]
data.predictability = ['a' if i==3 else 'b' for i in data.predictability] # which was which?
data.foreperiod = ['S' if i==0.65 else 'L' for i in data.foreperiod]
Q = ['TR','TL','BL','BR']
data.Quad = [Q[i-1] for i in data.Quad]
data.difficulty = [int((i-4)/2) for i in data.difficulty]
dumdata = pd.get_dummies(data,prefix=['quad','pred','forep']\
                          ,drop_first=False)


#######3 REGRESSION ###########
X_data = dumdata.drop(['score','reaction_time','TP','TN','FP','FN','difficulty','pred_a',\
                       'pred_b'],axis=1)
y_data = dumdata['reaction_time']
X2 = sm.add_constant(X_data)
mod = sm.OLS(y_data,X2)
estreg = mod.fit()
print(estreg.summary())





########## HEATMAP #############

#heatmap function from  https://gist.github.com/Kautenja/f9d6fd3d1dee631200bc11b8a46a76b7
def heatmap(data, size_scale=500, figsize=(10, 9), cmap=mpl.cm.RdBu):
    """
    Build a heatmap based on the given data frame (Pearson's correlation).
    Args:
        data: the dataframe to plot the cross-correlation of
        size_scale: the scaling parameter for box sizes
        figsize: the size of the figure
    Returns:
        None
    Notes:
        based on -
        https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    """
    # copy the data before mutating it
    data = data.copy()
    # change datetimes and timedelta to floating points
    for column in data.select_dtypes(include=[np.datetime64, np.timedelta64]):
        data[column] = data[column].apply(lambda x: x.value)
 
    # calculate the correlation matrix
    data = data.corr()
    data = pd.melt(data.reset_index(), id_vars='index')
    data.columns = ['x', 'y', 'value']
    x = data['x']
    y = data['y']
    # the size is the absolut value (correlation is on [-1, 1])
    size = data['value'].abs()
    norm = (data['value'] + 1) / 2

    fig, ax = plt.subplots(figsize=figsize)

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

    im = ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s', # Use square as scatterplot marker
        c=norm.apply(cmap)
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    # move the points from the center of a grid point to the center of a box
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    # move the ticks to correspond with the values
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    # move the starting point of the x and y axis forward to remove 
    # extra spacing from shifting grid points
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # add a color bar legend to the plot
    bar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.outline.set_edgecolor('grey')
    bar.outline.set_linewidth(1)
    bar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])

heatmap(dumdata)