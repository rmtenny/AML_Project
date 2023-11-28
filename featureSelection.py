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

# Load Data ------------------------------------------------------
# df = pd.read_csv('data/train_dataset.csv', index_col = 0)
# df_val = pd.read_csv('data/val_dataset.csv', index_col = 0)

# df = pd.read_csv('data/train_dataset_top.csv', index_col = 0)
# df_val = pd.read_csv('data/val_dataset_top.csv', index_col = 0)

# df.sort_values('mvel1',ascending=False).groupby('DATE').head(1000).reset_index(drop=True)
# df_val.sort_values('mvel1',ascending=False).groupby('DATE').head(1000).reset_index(drop=True)

df = pd.read_csv('data/train_dataset_bot.csv', index_col = 0)
df_val = pd.read_csv('data/val_dataset_bot.csv', index_col = 0)

df = df.dropna(subset = 'RET')
df_val = df_val.dropna(subset = 'RET')

y_train = df["RET"]
X_train = df.drop(columns=["DATE","permno","RET"])
y_val = df_val["RET"]
X_val = df_val.drop(columns=["DATE","permno","RET"])

# Define out of sample R2 -------------------------------------------
def R_oss(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    numer = np.dot((true-pred),(true-pred))
    denom = np.dot(true,true)
    frac = numer/denom
    return 1-frac

# # Test TSNE ----------------------------------
# tsne = TSNE(n_components=3, perplexity=30, n_iter=500, random_state=42)
# X_tsne = tsne.fit_transform(X_train)
# X_val_tsne = tsne.fit_transform(X_val) 
# linFitTsne = LinearRegression().fit(X_tsne, y_train)
# # r_squared_tsne = r2_score(y_val, linFitTsne.predict(X_val_tsne))
# r_squared_tsne = R_oss(y_val, linFitTsne.predict(X_val_tsne))
# adjusted_r_squared_tsne = 1 - (1-r_squared_tsne)*(len(y_val)-1)/(len(y_val)-X_val_tsne.shape[1]-1)
# print("R^2 with {} components tsne: {}".format(3,r_squared_tsne))

# Determine Best PCA model ----------------------------
vars = []
r2s = []
ar2s = []
r2s_tsne = []
ar2s_tsne = []
mses = []
nums = range(50,150,1)
for num in nums:
    pca = PCA(n_components=num)
    pca.fit(X_train,y_train)
    X_train_new = pca.transform(X_train)
    X_test_new = pca.transform(X_val)
    var = (sum(pca.explained_variance_ratio_))
    linFit = LinearRegression().fit(X_train_new, y_train)
    # r_squared = r2_score(y_val, linFit.predict(X_test_new))
    r_squared = R_oss(y_val, linFit.predict(X_test_new))
    adjusted_r_squared = 1 - (1-r_squared)*(len(y_val)-1)/(len(y_val)-X_test_new.shape[1]-1)
    y_val_per = linFit.predict(X_test_new)
    mse = mean_squared_error(y_val_per,y_val)
    print("Adjusted R^2 with {} components: {}".format(num,adjusted_r_squared))
    print("MSE {} components: {}".format(num,mse))
    print("R^2 with {} components: {}".format(num,r_squared))
    vars.append(var)
    r2s.append(r_squared)
    ar2s.append(adjusted_r_squared)
    mses.append(mse)
    print("Cumulative variance explained by {} components is {}".format(num,var)) #cumulative sum of variance explained with [n] features
plt.plot(nums,vars, label='variance')
plt.plot(nums,r2s, label='r2')
plt.plot(nums,ar2s, label='ar2')
# plt.plot(nums,r2s_tsne, label='r2_tsne')
# plt.plot(nums,ar2s_tsne, label='ar2 tsne')
plt.plot(nums,mses,label='MSE')

plt.title('PCA')
plt.xlabel("num PCA Components")
plt.ylabel("Explained Variance")
plt.legend()
plt.show()

## Print best PCA results --------------------------------
print('max R2:{}, at {}'.format(max(r2s),r2s.index(max(r2s))))

