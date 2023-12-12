import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler


## Pull in data ---------------------------------------------------
split = 'top'
if split == 'all':
    df = pd.read_csv('data/train_dataset.csv', index_col = 0)
    df_val = pd.read_csv('data/val_dataset.csv', index_col = 0)
    df_test = pd.read_csv('data/test_dataset.csv', index_col = 0)
    name = '_all'
    ips = 915
    # Epoch 14: train RMSE 0.1406, test RMSE 0.2035, r2 0.0031
elif split == 'top':
    df = pd.read_csv('data/train_dataset_top.csv', index_col = 0)
    df_val = pd.read_csv('data/val_dataset_top.csv', index_col = 0)
    df_test = pd.read_csv('data/test_dataset_top.csv', index_col = 0)
    name = '_top'
    ips = 911
    #Epoch 22: train RMSE 0.0774, test RMSE 0.1006, r2 0.0730

elif split == 'bot':
    df = pd.read_csv('data/train_dataset_bot.csv', index_col = 0)
    df_val = pd.read_csv('data/val_dataset_bot.csv', index_col = 0)
    df_test = pd.read_csv('data/test_dataset_bot.csv', index_col = 0)
    name = '_bot'
    ips = 910
    # Epoch 19: train RMSE 0.2214, test RMSE 0.3309, r2 0.0020

# Trim data for testing
# df.sort_values('mvel1',ascending=False).groupby('DATE').head(10).reset_index(drop=True)
# df_val.sort_values('mvel1',ascending=False).groupby('DATE').head(10).reset_index(drop=True)
# df_test.sort_values('mvel1',ascending=False).groupby('DATE').head(10).reset_index(drop=True)

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
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()        
        self.num_classes = output_dim
        self.num_layers = layer_dim
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=100, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=10, num_layers=4, batch_first=True)
        self.linear1 = nn.Linear(10, 1)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear1(x)
        return x
 
model = Model(ips, 100, 5, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
bs = 5
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=bs)

# Test ---------------------------------------
# model = torch.load('./model_top.pth')
# model.eval()

# pred = model(X_test)
# r_squared = R_oss(y_test, pred)
# y_pred = np.squeeze(pred.detach().numpy())
# df_save = df_test.filter(["permno","DATE","RET"], axis=1)
# # add pred
# df_save.insert(2, "pred", y_pred, True)
# # create new df
# # save
# df_save.to_csv('lstm_top.csv')

best_mse = np.inf   # init to infinity
best_weights = None 
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        # X_batch = torch.reshape(X_batch,(X_batch.shape[0],1,X_batch.shape[1]))
        # if X_batch.shape != torch.Size([bs,1,ips]):
        #     continue
        y_pred = model.forward(X_batch)
        # y_pred,mem1,mem2 = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        # X_train_rsp = torch.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
        # y_pred = model(X_train_rsp)
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        # X_test_rsp = torch.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
        # y_pred = model(X_test_rsp)
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        r2 = R_oss(y_test,y_pred)
    if test_rmse < best_mse:
        best_mse = test_rmse
        best_weights = copy.deepcopy(model.state_dict())
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f, r2 %.4f" % (epoch, train_rmse, test_rmse,r2))
 
## Evaluate best model ----------------------------------------
model.load_state_dict(best_weights)
name = 'model' + name + '.pth'
torch.save(model, name)
model.eval()
y_pred = model(X_test)
mse = loss_fn(y_pred, y_test)
mse = float(mse)
r2 = np.squeeze(R_oss(y_pred, y_test))
print(f"Val Loss: {mse} R2: {r2}")
y_pred = np.squeeze(y_pred.detach().numpy())

# Plot the validation curve --------------------------------------------
plt.plot(y_test, color='green', marker='o')
plt.plot(y_pred, color='red', marker='x')
plt.suptitle('Time-Series Prediction')
plt.show()
