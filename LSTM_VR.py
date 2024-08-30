import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm  # Standard tqdm import for non-Jupyter environments
from sklearn.preprocessing import MinMaxScaler

# Define data root directory
data_dir = "E:\\Users\\John\\Documents\\ReinfLe\\Practical DRL\\Ch3\\John_VR_Study\\New_OutputData_x"
path = "New_OutputData_x"
batch_size = 1024

def prepare_data(data_dir, lookback):
    label_scalers = {}
    train_x = []
    train_y = []
    test_x = {}
    test_y = {}

    # Use tqdm for progress indication
    for file in tqdm(os.listdir(data_dir)):  # Use standard tqdm here
        # Skip files that don't match criteria
        if not file.endswith(".csv") or file == "pjm_hourly_est.csv":
            continue

        # Store csv file in a Pandas DataFrame
        df = pd.read_csv(os.path.join(data_dir, file))
        df.columns = ['time', 'pos', 'acc', 'ref']
        df = df[['ref', 'time', 'pos', 'acc']]

        # Check if the dataset is large enough
        if len(df) <= lookback:
            # print(f"Skipping {file} due to insufficient data length ({len(df)} rows).")
            continue

        # Scaling the input data
        sc = MinMaxScaler()
        label_sc = MinMaxScaler()
        data = sc.fit_transform(df.values)
        
        # Obtaining the Scale for the labels (usage data) so that output can be re-scaled to actual value during evaluation
        label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
        label_scalers[file] = label_sc

        # Define lookback period and split inputs/labels
        inputs = np.zeros((len(data) - lookback, lookback, df.shape[1]))
        labels = np.zeros(len(data) - lookback)

        for i in range(lookback, len(data)):
            inputs[i - lookback] = data[i - lookback:i]
            labels[i - lookback] = data[i, 0]

        inputs = inputs.reshape(-1, lookback, df.shape[1])
        labels = labels.reshape(-1, 1)

        # Split data into train/test portions and combine all data from different files into a single array
        test_portion = int(0.1 * len(inputs))
        if len(train_x) == 0:
            train_x = inputs[:-test_portion]
            train_y = labels[:-test_portion]
        else:
            train_x = np.concatenate((train_x, inputs[:-test_portion]))
            train_y = np.concatenate((train_y, labels[:-test_portion]))
        
        test_x[file] = inputs[-test_portion:]
        test_y[file] = labels[-test_portion:]

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, test_x, test_y, label_scalers

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

#class GRUNet(nn.Module):
    #def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    #def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
    
def train(train_loader, learn_rate, hidden_dim=50, EPOCHS=100, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 3
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    LossEpoch = []
    TotalLossEpoch = []

    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()  # Use time.perf_counter() for more precise timing
        h = model.init_hidden(batch_size)
        avg_loss = 0.0
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss / counter))
        
        current_time = time.perf_counter()  # Use time.perf_counter() for more precise timing
        # print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        # print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
        LossEpoch.append(avg_loss / counter)
        TotalLossEpoch.append(avg_loss / len(train_loader))
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model, LossEpoch, TotalLossEpoch

def plot_losses(lstm_losses, gru_losses, lookback, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(lstm_losses, label='LSTM Loss')
    plt.plot(gru_losses, label='GRU Loss')
    plt.title(f'Loss per Epoch (Lookback={lookback})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot with a specific filename in the output directory
    plt.savefig(os.path.join(output_dir, f'loss_plot_lookback_{lookback}.png'))
    plt.close()

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.perf_counter()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(abs(outputs[i])+abs(targets[i])))
    return 200*sMAPE/len(outputs)

lookback_values = range(50, 100, 10)
lr = 0.001  # Learning rate

# Directory to save plots
plot_output_dir = "E:\\Users\\John\\Documents\\ReinfLe\\Practical DRL\\Ch3\\John_VR_Study\\LSTM_GRU_Plots"

for lookback in lookback_values:
    # Prepare data with the current lookback
    train_loader, test_x, test_y, label_scalers = prepare_data(data_dir, lookback)

    # Train the GRU model
    gru_model, gru_average_loss_epoch, gru_total_loss_epoch = train(train_loader, lr, model_type="GRU")

    # Train the LSTM model
    lstm_model, lstm_average_loss_epoch, lstm_total_loss_epoch = train(train_loader, lr, model_type="LSTM")

    # Print initial and final losses for GRU model
    print(f"\n*****LOOKBACK = {lookback}*******")
    print(f"GRU initial loss: {gru_total_loss_epoch[0]:.4f}")
    print(f"GRU final loss: {gru_total_loss_epoch[-1]:.4f}")

    # Print initial and final losses for LSTM model
    print(f"LSTM initial loss: {lstm_total_loss_epoch[0]:.4f}")
    print(f"LSTM final loss: {lstm_total_loss_epoch[-1]:.4f}")

    # Plotting the losses
    plot_losses(lstm_average_loss_epoch, gru_average_loss_epoch, lookback, plot_output_dir)
