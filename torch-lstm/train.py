import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMRegressor
from config import LSTM_PARAMS, BATCH_SIZE, EPOCHS
from process_data import preprocess_data, cut_y, Y_NAME


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # prepare data
    SPLIT_INDEX = 2000
    raw_data = pd.read_csv(os.path.join('..', 'data', 'smart-home', 'train.csv'))
    train_dataset, validate_dataset = raw_data[:SPLIT_INDEX], raw_data[SPLIT_INDEX:]

    train_x, train_y = cut_y(train_dataset, Y_NAME)
    validate_x, validate_y = cut_y(validate_dataset, Y_NAME)

    train_x = preprocess_data(train_x)
    validate_x = preprocess_data(validate_x)

    train_tdset = TensorDataset(torch.from_numpy(train_x.values), torch.from_numpy(train_x.values))
    train_loader = DataLoader(train_tdset, batch_size=BATCH_SIZE, shuffle=False)

    validate_tdset = TensorDataset(torch.from_numpy(validate_x.values), torch.from_numpy(validate_y.values))
    validate_loader = DataLoader(validate_tdset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize training objects
    model = LSTMRegressor(*LSTM_PARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
    loss_fun = nn.MSELoss()

    # start training
    model.train()

    for epoch in range(EPOCHS):
        for x, y in train_loader:
            hidden = model.init_hidden(len(x))
            optimizer.zero_grad()
            outputs, hidden = model(x, hidden)

            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        print("Epoch: %d, Loss: %f" % (epoch, loss.item()))


    # evaluate on validation set

    # save the model
    state_dict = model.state_dict()
    torch.save(state_dict, "model.tar")