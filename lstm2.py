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
df.sort_values('mvel1',ascending=False).groupby('DATE').head(100).reset_index(drop=True)
df_val.sort_values('mvel1',ascending=False).groupby('DATE').head(100).reset_index(drop=True)
df_test.sort_values('mvel1',ascending=False).groupby('DATE').head(100).reset_index(drop=True)

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

mm = MinMaxScaler()
y_trans_train = mm.fit_transform(y_train_np.reshape(-1, 1))
y_trans_val = mm.fit_transform(y_val_np.reshape(-1, 1))
y_trans_test = mm.fit_transform(y_test_np.reshape(-1, 1))

# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)
    
X_train_np, y_trans_train = split_sequences(X_train_np, y_trans_train, 10, 5)
X_val_np, y_trans_val = split_sequences(X_val_np, y_trans_val, 10, 5)
X_test_np, y_trans_test = split_sequences(X_test_np, y_trans_test, 10, 5)


X_train = Variable(torch.tensor(X_train_np, dtype=torch.float32))
y_train = Variable(torch.tensor(y_trans_train, dtype=torch.float32))
X_val = Variable(torch.tensor(X_val_np, dtype=torch.float32))
y_val = Variable(torch.tensor(y_trans_val, dtype=torch.float32))
X_test = Variable(torch.tensor(X_test_np, dtype=torch.float32))
y_test = Variable(torch.tensor(y_trans_test, dtype=torch.float32))
 
# X_train = torch.reshape(X_train,   
#                         (X_train.shape[0], 10, 
#                         X_train.shape[2]))
# X_val = torch.reshape(X_val,  
#                     (X_val.shape[0], 10, 
#                     X_val.shape[2])) 

def R_oss(true,pred):
    true = true.detach().numpy()
    pred = pred.detach().numpy()
    numer = np.dot((true-pred).T,(true-pred))
    denom = np.dot(true.T,true)
    frac = numer/denom
    return 1-frac

# train-test split for time series
# train_size = int(len(X_train))
# test_size = len(X_test)


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()        
        self.num_classes = output_dim
        self.num_layers = layer_dim
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.dropout = nn.Dropout(p = 0.2)
        
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=0.2)

        self.linear1 = nn.Linear(self.hidden_size, 25)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(25, self.num_classes)
    # def forward(self, x, mem1, mem2):
    def forward(self, x):
        h_0 = torch.squeeze(Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()))
        c_0 = torch.squeeze(Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()))

        res, (h_out, _) = self.lstm1(x, (h_0, c_0))
        # h_out = h_out.view(-1, self.hidden_size)
        h_out = h_out[self.num_layers - 1,:,:]
        out = self.relu(h_out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out

model = Model(ips, 100, 4, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
# bs = 256
# loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=bs)

# Test ---------------------------------------
# model = torch.load('./model_top.pth')
# model.eval()c

# pred = model(X_test)
# r_squared = R_oss(y_test, pred)
# pred = np.squeeze(pred.detach().numpy())
# df_save = df_test.filter(["permno","DATE","RET"], axis=1)
# # add pred
# df_save.insert(2, "pred", pred, True)
# # create new df
# # save
# df_save.to_csv('lstm_top.csv')

best_mse = np.inf   # init to infinity
best_weights = None 
n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Validation
    model.eval()
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
name = 'model' + name + '.pth'
torch.save(model, name)
model.eval()
y_pred = model(X_test)
mse = loss_fn(y_pred, y_test)
mse = float(mse)
r2 = np.squeeze(R_oss(y_pred, y_test))
print(f"Val Loss: {mse} R2: {r2}")

# Plot the validation curve --------------------------------------------
plt.plot(y_test, color='green', marker='o')
plt.plot(y_pred, color='red', marker='x')
plt.suptitle('Time-Series Prediction')
plt.show()
