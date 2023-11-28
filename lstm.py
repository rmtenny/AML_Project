import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import copy

## Pull in data ---------------------------------------------------
df = pd.read_csv('data/train_dataset.csv', index_col = 0)
df_val = pd.read_csv('data/val_dataset.csv', index_col = 0)
df_test = pd.read_csv('data/test_dataset.csv', index_col = 0)

# df = pd.read_csv('data/train_dataset_top.csv', index_col = 0)
# df_val = pd.read_csv('data/val_dataset_top.csv', index_col = 0)
# df_test = pd.read_csv('data/test_dataset_top.csv', index_col = 0)

# df = pd.read_csv('data/train_dataset_bot.csv', index_col = 0)
# df_val = pd.read_csv('data/val_dataset_bot.csv', index_col = 0)
# df_test = pd.read_csv('data/test_dataset_bot.csv', index_col = 0)

# Trim data for testing
# df.sort_values('mvel1',ascending=False).groupby('DATE').head(5).reset_index(drop=True)
# df_val.sort_values('mvel1',ascending=False).groupby('DATE').head(5).reset_index(drop=True)
# df_test.sort_values('mvel1',ascending=False).groupby('DATE').head(5).reset_index(drop=True)

# Make sure no nan exist
df = df.dropna(subset = 'RET')
df_val = df_val.dropna(subset = 'RET')
df_test = df_test.dropna(subset = 'RET')

y_train = df["RET"]
X_train = df.drop(columns=["DATE","permno","RET"])
y_val = df_val["RET"]
X_val = df_val.drop(columns=["DATE","permno","RET"])
y_test = df_test["RET"]
X_test = df_test.drop(columns=["DATE","permno","RET"])

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
 
def R_oss(true,pred):
    true = true.detach().numpy()
    pred = pred.detach().numpy()
    numer = np.dot((true-pred).T,(true-pred))
    denom = np.dot(true.T,true)
    frac = numer/denom
    return 1-frac

# train-test split for time series
train_size = int(len(X_train))
test_size = len(X_test)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=915, hidden_size=100, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=10, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(10, 1)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear1(x)
        return x
 
model = Model()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=5)

best_mse = np.inf   # init to infinity
best_weights = None 
n_epochs = 25
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    # model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        r2 = R_oss(y_test,y_pred)
    if test_rmse < best_mse:
        best_mse = test_rmse
        best_weights = copy.deepcopy(model.state_dict())
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f, r2 %.4f" % (epoch, train_rmse, test_rmse,r2))
 
## Evaluate best model ----------------------------------------
model.load_state_dict(best_weights)
model.eval()
y_pred = model(X_test)
mse = loss_fn(y_pred, y_test)
mse = float(mse)
r2 = np.squeeze(R_oss(y_pred, y_test))
print(f"Val Loss: {mse} R2: {r2}")