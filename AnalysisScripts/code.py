# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:20:56 2020

@author: Samira

Taras thesis
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
import warnings
warnings.filterwarnings("ignore")

os.getcwd()
os.listdir()

# df1 = pd.read_csv('Dropbox\Tara\Data\PooledData4DiffLev data\PooledData4DiffLev.csv')
df1 = pd.read_csv('Data//PooledData4DiffLev.csv') 
df1.head()

data = df1[['score','predictability', 'difficulty','foreperiod','TP', 'TN', 'FP', 'FN']]

data.predictability = [0 if i==3 else 1 for i in data.predictability]
data.difficulty = [int((i-4)/2) for i in data.difficulty]
# data.foreperiod = [0 if i==.65 else 1 for i in data.foreperiod]

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

heatmap(data)

### Heatmap only for midlevels (difficulty==1,2) ###

# Question: Now, will score have a correlation with predictability? No!!
ndata = data[data['difficulty'].isin([1,2])]
heatmap(ndata)

################## Random Forest ###################

def rand_for(X,y):
    X_train, X_test, y_train, y_test = train_test_split\
        (X, y, test_size=0.33, random_state=42)
        
    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train)
    rfc_predict = rfc.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))
    
    # training set accuracy (to test overfitting)
    train_predict = rfc.predict(X_train)
    print("Accuracy:",metrics.accuracy_score(y_train, train_predict))
    
    return

# 1. y = score
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['score']
rand_for(X_data, y_data)  # 0.7515576, # 0.75384

# 2. y = True pos
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['TP']
rand_for(X_data, y_data) # 0.6744548286604362 # 0.651689708141321

# 3. y = TN
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['TN']
rand_for(X_data, y_data) # 0.5700934579439252 # 0.5940860215053764

# 3. y = TN
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['TN']
rand_for(X_data, y_data) # 0.5700934579439252 # 0.5940860215053764

# 4. y = FP
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['FP']
rand_for(X_data, y_data) # 0.9143302180685359 # 0.9143625192012289

# 5. y = FP
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['FN']
rand_for(X_data, y_data) # 0.8434579439252337 # 0.8452380952380952


# Same work, only for mid-levels
mid_data = data.loc[data['difficulty'].isin([1,2])]
mid_data.head()

# heatmap
heatmap(mid_data)

X_data = mid_data[['predictability', 'difficulty','foreperiod']]
y_data = mid_data['score']
rand_for(X_data, y_data)


################ Logistic Regression ###############

# sklearn logis reg not able to find p-value

def log_reg(X,y):
    X_train, X_test, y_train, y_test = train_test_split\
        (X, y, test_size=0.33, random_state=42)
        
    lgc = LogisticRegression(random_state=0).fit(X, y)
    lgc_predict = lgc.predict(X_test)
    print("test accuracy:",metrics.accuracy_score(y_test, lgc_predict))
    print('coeficients:',lgc.coef_)
    
    # training set accuracy (to test overfitting)
    train_predict = lgc.predict(X_train)
    print("train accuracy:",metrics.accuracy_score(y_train, train_predict))
    
    return

# 1. y = score
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['score']
log_reg(X_data, y_data)

# 2. y = True pos
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['TP']
log_reg(X_data, y_data)

# 3. y = TN
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['TN']
log_reg(X_data, y_data) 

# 3. y = TN
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['TN']
log_reg(X_data, y_data)

# 4. y = FP
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['FP']
log_reg(X_data, y_data)

# 5. y = FP
X_data = data[['predictability', 'difficulty','foreperiod']]
y_data = data['FN']
log_reg(X_data, y_data)


########### Logistic regression using statsmodels ############


# THIS CONTAINS ERROR at line train accuracy print
import statsmodels.api as sm
# important link: https://stats.stackexchange.com/questions/203740/logistic-regression-scikit-learn-vs-statsmodels

# Result: if just mid-level diff, foreperiod is significant

def logit(X,y):
    X_train, X_test, y_train, y_test = train_test_split\
        (X, y, test_size=0.33, random_state=42)
        
    est = sm.Logit(y_train, X_train)
    est2 = est.fit()
    print(est2.summary())

    # training set accuracy (to test overfitting)
    # train_predict = est2.predict(X_train)
    # print("train accuracy:",metrics.accuracy_score(y_train, train_predict))
    
    return

ndata = data[data['difficulty'].isin([1,2])]
X_data = ndata[['predictability', 'difficulty','foreperiod']]
y_data = ndata['score']
X2 = sm.add_constant(X_data)
logit(X2, y_data)


# Bara hame adama joda joda anjam bede bebin tasiri ruye corre predic. mizare?
# Kollan ham bara adama joda joda anjam bede
# ROC curve

# Add reaction time to X_data***********

# Logistic Regression (on data containing only the middle level diffs?)

# Since X's don't have correlation, we can do "Causal analysis"****************

# REMOVE outliers and run again ***********

# Regress/likelihood reaction time as y ****** 
