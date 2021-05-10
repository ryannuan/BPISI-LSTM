# 利用10个特征预测window_forward天的感染率

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import BPISI_LSTM_DataLoad as preprocess
from scipy.stats import lognorm
import math
import os
import random


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def set_seed(seed):
    '''
    fix the seed.
    :param seed: seed.
    :return: None.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def loss_res(epochs, train_loss, val_loss, learning_rate, hidden_num):
    '''
    plot the loss trend of training and validating model.
    :param epochs: number of epoch.
    :param train_loss: training loss, list.
    :param val_loss: validation loss, list.
    :param learning_rate: lr.
    :param hidden_num: feature of hidden cells.
    :return:
    '''
    plt.clf()
    plt.plot(list(range(1, epochs+1)), train_loss, 'r', label='train loss')
    plt.plot(list(range(1, epochs+1)), val_loss, 'b', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig('../result/figure/India/3/BPISI_LSTM/loss_' + country +
                '_' + str(learning_rate) + '_' + str(hidden_num) + '_' + str(epochs) + '_' + str(window_backward) + '.png')


def rmse(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    return math.sqrt(((arr1 - arr2) ** 2).mean())


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(abs(y_true - y_pred))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(abs(y_true - y_pred) / y_true)


def confirm_prob(t, mu=7.2, sigma=np.log(15.1 / 7.2) / 1.6449):
    '''
    discretized log-normal distribution (from infected to confirmed).
    :param t: day.
    :param mu: the expectation.
    :param sigma: the variance.
    :return: discretized log-normal prob.
    '''
    temp = lognorm.cdf(t, s=sigma, loc=0, scale=mu) - lognorm.cdf(t-1, s=sigma, loc=0, scale=mu)
    temp /= lognorm.cdf(sample_num, s=sigma, loc=0, scale=mu)

    return temp


class lstm_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(lstm_model, self).__init__()
        self.input_size = input_size  # number of features OF input
        self.hidden_size = hidden_size  # number of features in the hidden state
        self.num_layers = num_layers  # number of layer
        self.output_size = output_size  # number of pred day
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器实例

    train_loss_list = []   # train_loss for each epoch
    val_loss_list = []   # val_loss for each epoch

    for i in range(epochs):
        train_loss = 0
        val_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            train_pred = model(seq)
            single_train_loss = loss_func(train_pred, labels)
            # print(single_train_loss)
            single_train_loss.backward()
            optimizer.step()
            # print(len(labels))
            train_loss += single_train_loss * len(labels)

        for seq, labels in val_loader:
            val_pred = model(seq)
            single_val_loss = loss_func(val_pred, labels)
            # print(single_val_loss)
            # print(len(labels))
            val_loss += single_val_loss * len(labels)

        train_loss_list.append(train_loss / len(train_y))
        val_loss_list.append(val_loss / len(val_y))

        print(f'epoch: {i:3}, train_loss: {train_loss / len(train_y):.6f}, val_loss: {val_loss / len(val_y):.6f}')

    loss_res(epochs, train_loss_list, val_loss_list, learning_rate, model.hidden_size)

    return model


def model_eval(model):
    '''
    predict the model for the train_val data.
    :param model: trained model.
    :return: predicted infection rates.
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


# def pred_infection_rate(model):



def model_test(model, test_X):
    '''
    :param model:
    :param test_X:
    :return:
    '''
    test_X_copy = test_X.copy()
    test_X_copy = torch.tensor(test_X_copy[-window_backward:, :], dtype=torch.float).reshape(-1, window_backward, n_features)
    pre_temp = model(test_X_copy).detach().numpy().reshape(window_forward).tolist()

    return pre_temp


def cal_infect_num(infect_num, infect_rate_pred):
    '''
    calculate the number of infected cases
    :param infect_num: number of historical infected cases.
    :param infect_rate_pred: estimated infected rates.
    :return: number of predicted infected cases.
    '''
    infect_num_copy = infect_num.copy()
    for i in range(window_forward+uncertain_day):
        temp = sum(infect_num_copy[-infect_day:]) * infect_rate_pred[i]
        infect_num_copy.append(temp[0])

    return infect_num_copy


def cal_confirmed_num(infect_num_copy):
    '''
    calculate the number of new daily confirmed cases.
    :param infect_num_copy: number of infected cases
    :return: number of predicted new daily confirmed cases.
    '''
    confirmed_delta_pred = []
    for i in range(window_forward):
        temp = np.array(infect_num_copy[:-(window_forward - i)])  # historical infection number
        probs = [confirm_prob(k) for k in range(len(temp))]  # the prob of intervals
        probs.reverse()
        confirmed_delta_pred.append(np.dot(temp, np.array(probs)))

    return confirmed_delta_pred


# 画出感染率曲线
def plot_infect_rate(infect_rate_pred, learning_rate, hidden_num):
    infect_rate_real = disease_pre[0]['infection_rate_10'][ind:-test_num].values.reshape(-1, 1)
    plt.clf()
    plt.plot(list(range(window_backward, window_backward+len(infect_rate_pred))),
             infect_rate_pred, c='b', label='BPISI_LSTM Fitting')
    plt.scatter(list(range(len(infect_rate_real))), infect_rate_real, c='r', label='Real', s=5)
    plt.legend(['BPISI_LSTM Fitting', 'Real'], loc="upper right")
    plt.savefig('../result/figure/India/3/BPISI_LSTM/infection_rate' + country +
                '_' + str(learning_rate) + '_' + str(hidden_num) + '_' + str(epochs) + '_' + str(window_backward) + '.png')


def plot_confirmed(test_pred, learning_rate, hidden_num):
    real = feature_country['confirmed_delta'].values.reshape(-1, 1)
    test_pred = np.array(test_pred).reshape(-1, 1)
    test_pred_merge = np.concatenate((real[-31].reshape(-1, 1), test_pred))

    plt.clf()
    plt.scatter(list(range(len(real))), real, s=5, c='r', label='Real Count')
    plt.plot(list(range((len(real)-(30+1)), len(real))), test_pred_merge,
             c='orange', label='BPISI_LSTM prediction')
    plt.legend(loc='best')
    plt.title('Number of confirmed for ' + country + ' (' + str(window_forward) + ' days)')
    plt.vlines(len(real)-30, min(real), max(real), colors="grey", linestyles="dashed")
    plt.savefig('../result/figure/India/3/BPISI_LSTM/confirmed_delta_' + country +
                '_' + str(learning_rate) + '_' + str(hidden_num) + '_' + str(epochs) + '_' + str(window_backward) + '.png')


if __name__ == '__main__':
    os.chdir('../../')
    set_seed(100)

    epochs = 380
    batch_size = 8
    learning_rate = 0.0001
    window_backward = 10   # 历史窗口
    window_forward = 3   # 预测窗口
    uncertain_day = 15   # 不确定天数
    infect_day = 10   # 感染者可传染的天数
    test_num = 30 + uncertain_day
    n_features = 10


    country = 'India'


    disease = preprocess.disease()   # disease-related data
    mobility = preprocess.mobility(country)   # mobility info
    feature_country, ind = preprocess.merge_hist(mobility, disease, country, mask_ind=None)   # merged features


    train_val_data = feature_country.iloc[:-test_num,:]
    # train_val_data_scaled, label, scaler, min_max_scaler = preprocess.z_score_scale(train_val_data)
    # hybrid normalize all the features
    train_val_data_scaled, label, scaler, min_max_scaler = preprocess.hybrid_scale(train_val_data)
    # generate the sequence for train and val
    train_val_data_scaled_sample = preprocess.series_to_supervised(train_val_data_scaled, window_backward)
    train_X, train_y, val_X, val_y, train_val_X, train_val_y = preprocess.data_split(
        train_val_data_scaled_sample, label, window_backward, window_forward, n_features)
    sample_num = disease[country].shape[0]


    train = Data.TensorDataset(train_X, train_y)
    train_loader = Data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    val = Data.TensorDataset(val_X, val_y)
    val_loader = Data.DataLoader(dataset=val, batch_size=1, shuffle=False)
    train_val = Data.TensorDataset(train_val_X, train_val_y)
    train_val_loader = Data.DataLoader(dataset=train_val, batch_size=1, shuffle=False)


    LSTM = lstm_model(input_size=n_features, hidden_size=16, num_layers=1, output_size=window_forward, batch_size=batch_size)
    LSTM = model_train(LSTM)


    train_val_pred = model_eval(LSTM)   # 训练集和验证集的预测值
    train_val_pred = np.array([j for i in train_val_pred for j in i]).reshape(-1, 1)
    train_val_pred = scaler.inverse_transform(train_val_pred)


    # # 预测不准的15天的感染率
    # df = preprocess.merge_hist(mobility, disease, country, 30)
    # data_set = [df.iloc[:-(30+ind), 2:-1].values.reshape(-1, n_features) for ind in [15, 12, 9, 6, 3]]
    # test_X_set = [min_max_scaler.transform(data) for data in data_set]
    # test_pred_15 = [model_test(LSTM, data_scaled) for data_scaled in test_X_set]
    # infect_rate_pred_15 = [np.array(test_pred).reshape(-1,1) for test_pred in test_pred_15]
    # # inverse transform
    # infect_rate_pred_15 = [scaler.inverse_transform(infect_rate_pred) for infect_rate_pred in infect_rate_pred_15]
    # infect_rate_pred_15 = np.concatenate(infect_rate_pred_15, axis=0)
    #
    #
    # disease_pre = pd.read_csv('../result/disease_pre/India/3/' + country + '_' + str(30) + '.csv', header=0)
    # # historical number of infected cases
    # infect_num = disease_pre['infection_delta_est'][:-(30 + 15)].values.tolist()
    # # predict the number of infected cases using ISI
    # infect_num_pred_set = cal_infect_num(infect_num, infect_rate_pred_15)


    inds = [30, 27, 24, 21, 18, 15, 12, 9, 6, 3]
    dfs = [preprocess.merge_hist(mobility, disease, country, ind) for ind in inds]
    # historical features
    data_set = [[df.iloc[:-(ind+uncertain_day-3*i), 2:-1].values.reshape(-1, n_features) for i in range(6)]
                for ind, df in zip(inds, dfs)]
    # data normalization
    test_X_set = [[min_max_scaler.transform(data[i]) for i in range(6)] for data in data_set]
    # predict the infection rate
    test_pred_set = [[model_test(LSTM, data_scaled[i]) for i in range(6)] for data_scaled in test_X_set]
    test_pred_set = [[z for j in i for z in j] for i in test_pred_set]
    infect_rate_pred_set = [np.array(test_pred).reshape(-1,1) for test_pred in test_pred_set]
    # inverse transform
    infect_rate_pred_set = [scaler.inverse_transform(infect_rate_pred) for infect_rate_pred in infect_rate_pred_set]


    disease_pre = [pd.read_csv('../result/disease_pre/India/3/' + country + '_' + str(ind) + '.csv', header=0) for ind in inds]
    # historical number of infected cases
    infect_num_set = [df['infection_delta_est'][:-(ind + uncertain_day)].values.tolist() for ind, df in zip(inds, disease_pre)]
    # predict the number of infected cases using ISI
    infect_num_pred_set = [cal_infect_num(infect_num, infect_rate_pred[-(window_forward+uncertain_day):])
                  for infect_num, infect_rate_pred in zip(infect_num_set, infect_rate_pred_set)]


    # historical number of confirmed cases
    confirmed_delta_real = feature_country['confirmed_delta'][-30:].values.tolist()
    confirmed_delta_real = [confirmed_delta_real[i * 3:(i + 1) * 3] for i in range(10)]
    # calculate the number of confirmed cases using BP
    confirmed_delta_pred = [cal_confirmed_num(infect_num_pred) for infect_num_pred in infect_num_pred_set]
    # confirmed_delta_pred = [j for i in confirmed_delta_pred for j in i]
    pred_rmse = np.mean([rmse(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(10)])
    pred_mape = np.mean([mape(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(10)])
    pred_mae = np.mean([mae(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(10)])


    plot_infect_rate(train_val_pred, learning_rate, LSTM.hidden_size)
    plot_confirmed(confirmed_delta_pred, learning_rate, LSTM.hidden_size)


    with open('../result/figure/India/3/BPISI_LSTM/' + country + '_model_res.txt', 'a') as file:
        file.write('hidden number {}, learning rate {}, epochs {}, batch size {}, '
                   'window_backward {}, window_forward {}, uncertain_day {}, infect_day {}, '
                   'LSTM_rmse {:d}, LSTM_mae {:d}, LSTM_mape {:.6f}'.format(
            LSTM.hidden_size, learning_rate, epochs, batch_size,
            window_backward, window_forward, uncertain_day, infect_day,
            round(pred_rmse), round(pred_mae), pred_mape))
        file.write('\n')


    # plt.clf()
    # plt.plot(range(len(train_val_pred)), train_val_pred)
    # plt.scatter(list(range(len(infect_rate_real))), infect_rate_real, c='r', label='Real', s=5)
    # plt.savefig('test.png')
    # plt.plot(range(len(train_val_pred), len(train_val_pred)+len(infect_rate_pred_set[0][-5:])), infect_rate_pred_set[0][-5:])
    # plt.savefig('test.png')




