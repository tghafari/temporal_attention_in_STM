# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:54:50 2021

@author: Ali_Rahi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from google.colab import files
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
import pingouin as pg
import outdated
OUTDATED_IGNORE=1

#uploaded = files.upload()
df = pd.read_csv('DataFrameQuad_RT_TP.csv')
print(df.columns)

import numpy.matlib
#rd_label = np.matlib.repmat(np.array(['ARU1', 'ARU2', 'ARU3', 'ARU4', 'ARU5', 'RU1', 'RU2', 'RU3', 'RU4', 'RU5', 'ARD1', 'ARD2', 'ARD3', 'ARD4', 'ARD5', 'RD1', 'RD2', 'RD3', 'RD4', 'RD5']), 13, 1).flatten()
#df['rd_label'] = rd_label
#print(df)

# df.dropna(axis=0,how='any',inplace=True)

#ttest_ind(rdf, ardf)

###############################################################################
Anova_results = AnovaRM(df, 'nanmean_RT_TP', 'Subject_code', within=['foreperiod','predictability','difficulty'], aggregate_func = 'mean').fit()

print(Anova_results)

p_res = pg.pairwise_ttests(data = df, dv = 'nanmean_RT_TP', within=['Merged'],subject='Subject_code', padjust= 'bonf') # Cannot have morethan 2 elements

p_res.to_excel('DataFrameQuad_RT_TP_pairwise.xlsx')
###############################################################################


# pg.plot_paired(data = df, dv = 'nanmean_score', within = 'Merged', subject='Subject_code', boxplot = False, colors = ['Green', 'Blue', 'Red', 'Orange', 'Yellow'])

# multipletests(Anova_results.anova_table['Pr > F'])

# Anova_results.anova_table.keys()

# import scikit_posthocs as sp
# sp.posthoc_ttest(df, val_col='RT', group_col='rd_label', p_adjust='bonferroni')

# for key in A.keys():
#     print(key + '\n')
#     print(A[key][A[key]<0.05].index.values)
#     print('-------------------------')
    
# A.keys()

