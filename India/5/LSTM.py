# 利用10个特征预测window_forward天的确诊人数

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import LSTM_DataLoad as preprocess
import math
import os
import random


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def set_seed(seed):
    '''
    fix the seed
    :param seed: seed
    :return: None
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# 画出模型的效果图
def loss_res(epochs, train_loss, val_loss, learning_rate, hidden_num):
    plt.clf()
    plt.plot(list(range(1, epochs+1)), train_loss, 'r', label='train loss')
    plt.plot(list(range(1, epochs+1)), val_loss, 'b', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig('../result/figure/India/5/LSTM/loss_' + country +
                '_' + str(learning_rate) + '_' + str(hidden_num) + '_' + str(epochs) + '_' + str(window_backward) + '.png')


def model_res(test_pred, ind, learning_rate, hidden_num):
    real = disease[country]['confirmed_delta'].values.reshape(-1, 1)
    test_pred = test_pred.reshape(-1, 1)
    test_pred_merge = np.concatenate((train_val_pred[-1:], test_pred))

    plt.clf()
    plt.scatter(list(range(len(real))), real, s=5, c='r', label='Real Count')
    plt.plot(list(range(ind+window_backward, ind+window_backward+len(train_val_pred))), train_val_pred,
             c='b', label='LSTM fitting')
    plt.plot(list(range((len(real)-(test_num+1)), len(real))), test_pred_merge,
             c='orange', label='LSTM prediction')
    plt.legend(loc='best')
    plt.title('Number of confirmed for ' + country + ' (' + str(window_forward) + ' days)')
    plt.vlines(ind + window_backward, min(real), max(real), colors="grey", linestyles="dashed")
    # plt.vlines(len(real)-5, min(real), max(real), colors="grey", linestyles="dashed")
    plt.savefig('../result/figure/India/5/LSTM/confirmed_delta_' + country +
                '_' + str(learning_rate) + '_' + str(hidden_num) + '_' + str(epochs) + '_' + str(window_backward) + '.png')


def rescaled(x, minval, maxval):
    '''
    :param x:
    :param minval: the minimal number of historical confirmed cases
    :param maxval: the maximum number of historical confirmed cases
    :return: rescaled value
    '''
    x = np.array(x)
    return (maxval-minval) * x + minval


def rmse(arr1, arr2):
    return math.sqrt(((arr1 - arr2) ** 2).mean())


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(abs(y_true - y_pred))


def mape(y_true, y_pred):
    return np.mean(abs(y_true - y_pred) / y_true)


class lstm_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(lstm_model, self).__init__()
        self.input_size = input_size   # number of features OF input
        self.hidden_size = hidden_size   # number of features in the hidden state
        self.num_layers = num_layers   # number of layer
        self.output_size = output_size   # number of pred day
        self.batch_size = batch_size

        # (seq_len, batch, input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        input = input.transpose(0, 1)  # (seq_len, batch_size, n_feature)
        lstm_out, self.hidden_cell = self.lstm(input)  # lstm_out: (seq_len, batch_size, hidden_size)
        y_pred = self.fc(lstm_out)
        return y_pred[-1]


def model_train(model):
    '''
    train the model.
    :param model: model.
    :return: trained LSTM.
    '''
    model.train()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []   # train_loss for each epoch
    val_loss_list = []   # val_loss for each epoch

    for i in range(epochs):
        train_loss = 0
        val_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            train_pred = model(seq)

            single_loss = loss_func(train_pred, labels)
            single_loss.backward()
            optimizer.step()
            train_loss += single_loss * len(labels)

        for seq, labels in val_loader:
            val_pred = model(seq)
            single_loss = loss_func(val_pred, labels)
            val_loss += single_loss * len(labels)

        train_loss_list.append(train_loss / len(train_y))
        val_loss_list.append(val_loss / len(val_y))

        print(f'epoch: {i:3}, train_loss: {train_loss / len(train_y):.6f}, val_loss: {val_loss / len(val_y):.6f}')

    loss_res(epochs, train_loss_list, val_loss_list, learning_rate, model.hidden_size)

    return model


def model_eval(model):
    '''
    predict the model for the train_val data.
    :param model: trained model.
    :return: predicted number of confirmed cases.
    '''
    model.eval()
    train_val_pred = []
    train_val_real = []
    for (id,(seq, labels)) in enumerate(train_val_loader):
        # if is not the last sample
        if id != len(train_val_loader)-1:
            # select the first value as the prediction
            train_val_pred.append([model(seq).detach().numpy()[0][0]])
            train_val_real.append([labels.detach().numpy()[0][0]])
        # if is the last sample
        else:
            train_val_pred.append(model(seq).detach().numpy().reshape(-1).tolist())
            train_val_real.append(labels.detach().numpy().reshape(-1).tolist())

    return train_val_pred


def model_pred(model):
    test_pred_set = []
    test_real_set = []
    for i in range(1, 7):
        if i != 6:
            test_data = feature_country.iloc[-test_num:-(test_num-window_forward*i), 2:]
        else:
            test_data = feature_country.iloc[-(test_num-window_forward*i):, 2:]
        # 0-1 normalize the test_X
        test_data_scaled = scaler.transform(test_data.values.reshape(-1, n_features))
        # normalized train_val_test_X
        data_scaled = np.concatenate((train_val_data_scaled, test_data_scaled), axis=0)
        test_y = data_scaled[-window_forward:,0]   # take the latest window_forward days as the label
        test_X = torch.tensor(data_scaled[-(window_forward+window_backward):-window_forward,:], dtype=torch.float).reshape(
            -1, window_backward, n_features)   # take the previous days as the X
        pre_temp = model(test_X).detach().numpy()   # predict window_forward days
        test_pred_set.append(pre_temp.reshape(-1).tolist())
        test_real_set.append(test_y.tolist())
    return test_pred_set, test_real_set


if __name__ == '__main__':
    os.chdir('../../')
    set_seed(100)   # fix the seed


    batch_size = 8
    learning_rate = 0.0001   # lr
    epochs = 300
    window_backward = 10   # 历史窗口
    window_forward = 5   # 预测窗口
    test_num = 30   # 测试集的样本量
    n_features = 10   # number of features


    country = 'India'


    disease = preprocess.disease()   # disease-related data
    mobility = preprocess.mobility(country)   # mobility info
    feature_country, ind = preprocess.merge_hist(mobility, disease, country)   # merged features


    train_val_data = feature_country.iloc[:-test_num, 2:]   # train-validation data
    # 0-1 normalize all the features
    train_val_data_scaled, minval, maxval, scaler = preprocess.maxmin_scale(train_val_data)
    # generate the sequence for train and val
    train_val_data_scaled_sample = preprocess.series_to_supervised(
        train_val_data_scaled, seq_len=window_backward, n_out=window_forward)


    col_ind = list(range(window_backward * n_features))
    col_ind.extend([window_backward * n_features + i * n_features for i in range(window_forward)])   # sample + label
    train_val_data_scaled_sample = train_val_data_scaled_sample.iloc[:, col_ind]
    train_X, train_y, val_X, val_y, train_val_X, train_val_y = preprocess.data_split(
        train_val_data_scaled_sample, window_backward, window_forward, n_features)


    train = Data.TensorDataset(train_X, train_y)
    train_loader = Data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    val = Data.TensorDataset(val_X, val_y)
    val_loader = Data.DataLoader(dataset=val, batch_size=1, shuffle=False)
    train_val = Data.TensorDataset(train_val_X, train_val_y)
    train_val_loader = Data.DataLoader(dataset=train_val, batch_size=1, shuffle=False)


    LSTM = lstm_model(input_size=n_features, hidden_size=16, num_layers=1, output_size=window_forward, batch_size=batch_size)
    LSTM = model_train(LSTM)
    train_val_pred = model_eval(LSTM)
    train_val_pred = [j for i in train_val_pred for j in i]
    train_val_pred =  rescaled(np.array(train_val_pred).reshape(-1, 1), minval, maxval)


    test_pred_set, test_y_set= model_pred(LSTM)
    test_pred_0 = rescaled([i for pred in test_pred_set for i in pred], minval, maxval)
    # test_real = rescaled([i for real in test_y_set for i in real], minval, maxval)
    test_pred = [rescaled([i for i in pred], minval, maxval) for pred in test_pred_set]
    test_real = [rescaled([i for i in pred], minval, maxval) for pred in test_y_set]

    pred_rmse = np.mean([rmse(test_pred[i], test_real[i]) for i in range(6)])
    pred_mape = np.mean([mape(test_pred[i], test_real[i]) for i in range(6)])
    pred_mae = np.mean([mae(test_pred[i], test_real[i]) for i in range(6)])
    # LSTM_test_rmse = rmse(test_real, test_pred)
    # LSTM_test_mape = mape(test_real, test_pred)
    # LSTM_test_mae = mae(test_real, test_pred)


    model_res(test_pred_0, ind, learning_rate, LSTM.hidden_size)
    with open('../result/figure/India/5/LSTM/' + country + '_model_res.txt', 'a') as file:
        file.write('hidden number {}, learning_rate {}, epochs {}, window_backward {}, batch size {}, '
                   'test_rmse: {:d}, test_mae: {:d}, test_mape: {:.6f}'.format(
            LSTM.hidden_size, learning_rate, epochs, window_backward, batch_size,
            round(pred_rmse), round(pred_mae), pred_mape))
        file.write('\n')
