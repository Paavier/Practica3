# %%
#Autor: Pedro Javier Ortega Fernandez
#Titulación: Doble Grado Matemáticas e Ingeniería Informática

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# %%
def TN(dataframe, threadshold):
    return ((dataframe['gt'] == 0) & (dataframe['pred'] < threadshold)).sum()

# %%
def TP (dataframe, threadshold):
   return ((dataframe['gt'] == 1) & (dataframe['pred'] >= threadshold)).sum()

# %%
def FN (dataframe, threadshold):
    return ((dataframe['gt'] == 1) & (dataframe['pred'] < threadshold)).sum()

# %%
def FP (dataframe, threadshold):
    return ((dataframe['gt'] == 0) & (dataframe['pred'] >= threadshold)).sum()

# %%
def Accuracy (dataframe, threadshold):
    tp = TP(dataframe, threadshold)
    tn = TN(dataframe, threadshold)
    fp = FP(dataframe, threadshold)
    fn = FN(dataframe, threadshold)
    
    return (tp + tn)/(tp + tn + fp + fn)

# %%
def Precision(dataframe, threadshold):
    tp = TP(dataframe, threadshold)
    fp = FP(dataframe, threadshold)
    return tp/(tp + fp)

# %%
def Recall(dataframe, threadshold):
    tp = TP(dataframe, threadshold)
    fn = FN(dataframe, threadshold)
    return tp/(tp + fn)

# %%
def TNR(dataframe, threadshold):
    tn = TN(dataframe, threadshold)
    fp = FP(dataframe, threadshold)
    return tn/(tn + fp)

# %%
def FPR(dataframe, threadshold):
    fp = FP(dataframe, threadshold)
    tn = TN(dataframe, threadshold)
    return fp/(fp + tn)

# %%
def F1_score(dataframe, threadshold):
    precision = Precision(dataframe, threadshold)
    recall = Recall(dataframe, threadshold)
    return 2*(precision * recall)/(precision + recall)

# %%
def Auc (dataframe):
    return roc_auc_score(dataframe['gt'], dataframe['pred'])
