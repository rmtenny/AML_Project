from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
import os, sys, re
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('data/train_dataset.csv', index_col = 0)
df_val = pd.read_csv('data/val_dataset.csv', index_col = 0)
y_train = df["RET"]
X_train = df.drop(columns=["DATE","permno","RET"])
y_val = df["RET"]
X_val = df.drop(columns=["DATE","permno","RET"])

vars = []
r2s = []
ar2s = []
r2s_tsne = []
ar2s_tsne = []
nums = range(1,900,10)
for num in nums:
    pca = PCA(n_components=num)
    pca.fit(X_train,y_train)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_val)
    # tsne = TSNE(n_components=num, perplexity=30, n_iter=250, random_state=42)
    # X_tsne = tsne.fit_transform(X_train)
    # X_val_tsne = tsne.fit_transform(X_val) 
    var = (sum(pca.explained_variance_ratio_)*100)
    linFit = LinearRegression().fit(X_train_new, y_train)
    # linFitTsne = LinearRegression().fit(X_tsne, y_train)
    r_squared = r2_score(y_val, linFit.predict(X_test_new))
    adjusted_r_squared = 1 - (1-r_squared)*(len(y_val)-1)/(len(y_val)-X_test_new.shape[1]-1)
    # r_squared_tsne = r2_score(y_val, linFitTsne.predict(X_val_tsne))
    # adjusted_r_squared_tsne = 1 - (1-r_squared_tsne)*(len(y_val)-1)/(len(y_val)-X_val_tsne.shape[1]-1)
    print("Adjusted R^2 with {} components: {}".format(num,adjusted_r_squared))
    print("R^2 with {} components: {}".format(num,r_squared))
    # print("R^2 with {} components tsne: {}".format(num,r_squared_tsne))
    vars.append(var)
    r2s.append(r_squared)
    ar2s.append(adjusted_r_squared)
    # r2s_tsne.append(r_squared_tsne)
    # ar2s_tsne.append(adjusted_r_squared_tsne)
    print("Cumulative variance explained by {} components is {}".format(num,var)) #cumulative sum of variance explained with [n] features
plt.plot(nums,vars, label='variance')
plt.plot(nums,r2s, label='r2')
plt.plot(nums,ar2s, label='ar2')
# plt.plot(nums,r2s_tsne, label='ar2_tsne')
# plt.plot(nums,ar2s_tsne, label='ar2 tsne')

plt.title('Explained Variance')
plt.xlabel("num PCA Components")
plt.ylabel("Explained Variance")
plt.legend()


## Try to do PCA on just a single ticker and see if there is any correlation



