import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import joblib

data_dir = "E:\\Users\\John\\Documents\\ReinfLe\\Practical DRL\\Ch3\\VR_Study\\new_z_axis"

all_scalers = {'scalers': {}, 'label_scalers': {}}

label_scalers = {}
train_x = []
test_x = []
test_y = []

for file in tqdm(os.listdir(data_dir), desc="Processing Files", unit="file"):
    # Store csv file in a Pandas DataFrame
    df = pd.read_csv('{}/{}'.format(data_dir, file))
    df.columns = ['time', 'pos', 'acc', 'ref']
    df = df[['ref', 'time', 'pos', 'acc']]

    # Scaling the input data
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(df.values)

    # Obtaining the Scale for the labels(usage data)
    label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
    label_scalers[file] = label_sc

    # Define lookback period and split inputs/labels
    lookback = 90
    inputs = np.zeros((len(data) - lookback, lookback, df.shape[1]))
    labels = np.zeros(len(data) - lookback)

    for i in range(lookback, len(data)):
        inputs[i - lookback] = data[i - lookback:i]
        labels[i - lookback] = data[i, 0]
    inputs = inputs.reshape(-1, lookback, df.shape[1])
    labels = labels.reshape(-1, 1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.2 * len(inputs))
    if len(train_x) == 0:
        train_x = inputs[:-test_portion]
        train_y = labels[:-test_portion]
    else:
        train_x = np.concatenate((train_x, inputs[:-test_portion]))
        train_y = np.concatenate((train_y, labels[:-test_portion]))

    test_x.append(inputs[-test_portion:])
    test_y.append(labels[-test_portion:])

    # Save the scalers in the dictionary
    file_name_without_extension = os.path.splitext(file)[0]
    all_scalers['scalers'][file_name_without_extension] = sc
    all_scalers['label_scalers'][file_name_without_extension] = label_sc

# Save all the scalers in two separate files using joblib
scaler_dir = 'E:\\Users\\John\\Documents\\ReinfLe\\Practical DRL\\Ch3\\VR_Study\\scalers'  # You can change this directory
os.makedirs(scaler_dir, exist_ok=True)

joblib.dump(all_scalers['scalers'], os.path.join(scaler_dir, 'input_scalers.joblib'))
joblib.dump(all_scalers['label_scalers'], os.path.join(scaler_dir, 'label_scalers.joblib'))