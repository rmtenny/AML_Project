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
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tqdm
import copy

## Pull in data ---------------------------------------------------
# df = pd.read_csv('data/train_dataset.csv', index_col = 0)
# df_val = pd.read_csv('data/val_dataset.csv', index_col = 0)
# df_test = pd.read_csv('data/test_dataset.csv', index_col = 0)

df = pd.read_csv('data/train_dataset_top.csv', index_col = 0)
df_val = pd.read_csv('data/val_dataset_top.csv', index_col = 0)
df_test = pd.read_csv('data/test_dataset_top.csv', index_col = 0)

df.sort_values('mvel1',ascending=False).groupby('DATE').head(10).reset_index(drop=True)
df_val.sort_values('mvel1',ascending=False).groupby('DATE').head(10).reset_index(drop=True)
df_test.sort_values('mvel1',ascending=False).groupby('DATE').head(10).reset_index(drop=True)

# df = pd.read_csv('data/train_dataset_bot.csv', index_col = 0)
# df_val = pd.read_csv('data/val_dataset_bot.csv', index_col = 0)
# df_test = pd.read_csv('data/test_dataset_bot.csv', index_col = 0)

df = df.dropna(subset = 'RET')
df_val = df_val.dropna(subset = 'RET')
df_test = df_test.dropna(subset = 'RET')

y_train = df["RET"]
X_train = df.drop(columns=["DATE","permno","RET"])
y_val = df_val["RET"]
X_val = df_val.drop(columns=["DATE","permno","RET"])
y_test = df_test["RET"]
X_test = df_test.drop(columns=["DATE","permno","RET"])


## Build model ------------------------------------
# Convert to 2D PyTorch tensors
X_train_np = X_train.to_numpy() 
y_train_np = y_train.to_numpy() 
X_val_np = X_val.to_numpy() 
y_val_np = y_val.to_numpy() 
X_test_np = X_test.to_numpy() 
y_test_np = y_test.to_numpy() 

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)
 
# Define the model
model = nn.Sequential(
    nn.Linear(911, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
 
# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
 
for epoch in range(n_epochs):
    print('epoch {}'.format(epoch))
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
 
# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()