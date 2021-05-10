# 利用Back projection-ISI (BPISI)方法

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import lognorm
import math
import os


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
    # discretized log-normal distribution (from infected to confirmed).
    '''
    :param t: day.
    :param mu: the expectation.
    :param sigma: the variance.
    :return: discretized log-normal prob.
    '''
    temp = lognorm.cdf(t, s=sigma, loc=0, scale=mu) - lognorm.cdf(t-1, s=sigma, loc=0, scale=mu)
    temp /= lognorm.cdf(sample_num, s=sigma, loc=0, scale=mu)

    return temp


# 两参数指数函数拟合
def exp_func(x, infection_rate):
    '''
    :param x: the number of days including fitted and pred
    :param infection_rate: infection rates needed to be fitted
    :return:
    '''
    time_len = len(infection_rate)
    lr = LinearRegression()
    lr.fit(np.array(list(range(time_len))).reshape(-1,1), np.log(infection_rate))
    b = -lr.coef_[0]
    a = np.exp(lr.intercept_)

    return a * np.exp(-b * np.array(x))


# 去掉早期的样本点
def drop_sample(data):
    '''
    如果day t的新增确诊数量为0，且day t+1 and day t+2的新增确诊数量也为0，则去除day t的样本
    :param data: raw COVID-19 daily dataset released by Johns Hopkins University.
    :return: preprocessed data.
    '''
    confirmed_delta = data['confirmed_delta_real']
    ind = 0
    for i in range(len(confirmed_delta)-2):
        if ((confirmed_delta[i] != 0) & (confirmed_delta[i+1] != 0) & (confirmed_delta[i+2] != 0)):
            ind = i
            break
    data = data.iloc[ind:,:]

    return data


# 计算新增感染人数
def cal_infect_num(infect_num, infect_rate_pred):
    '''
    calculate the number of new daily infected cases.
    :param infect_num: number of historical new daily infected cases.
    :param infect_rate_pred: estimated new daily infected rates.
    :return: number of predicted new daily infected cases.
    '''
    infect_num_copy = infect_num.copy()
    for i in range(pred_day):
        temp = sum(infect_num_copy[-infect_day:]) * infect_rate_pred[i]
        infect_num_copy.append(temp)

    return infect_num_copy


# 计算新增确诊人数
def cal_confirmed_num(infect_num_copy):
    '''
    calculate the number of new daily confirmed cases.
    :param infect_num_copy: number of infected cases
    :return: number of predicted new daily confirmed cases.
    '''
    confirmed_delta_pred = []
    for i in range(window_forward):
        temp = np.array(infect_num_copy[:-(window_forward - i)])  # historical infection rates
        probs = [confirm_prob(k) for k in range(len(temp))]  # the prob of intervals
        probs.reverse()
        confirmed_delta_pred.append(np.dot(temp, np.array(probs)))

    return confirmed_delta_pred


def plot_confirmed(test_pred):
    # the real number of new daily confirmed cases
    real = dfs[0]['confirmed_delta_real'].values.reshape(-1, 1)
    test_pred = np.array(test_pred).reshape(-1, 1)
    # concate the last 31th day to keep the continuity of the line
    test_pred_merge = np.concatenate((real[-31].reshape(-1, 1), test_pred))

    plt.clf()
    plt.scatter(list(range(len(real))), real, s=5, c='r', label='Real Count')
    plt.plot(list(range((len(real)-(30+1)), len(real))), test_pred_merge,
             c='orange', label='BPISI prediction')
    plt.legend(loc='best')
    plt.title('Number of confirmed for ' + country + ' (' + str(window_forward) + ' days)')
    plt.vlines(len(real)-30, min(real), max(real), colors="grey", linestyles="dashed")
    plt.savefig('../result/figure/India/10/BPISI/confirmed_delta_' + country +
                '_' + '_result.png')


if __name__ == '__main__':
    os.chdir('../../')

    window_forward = 10   # 预测窗口
    uncertain_day = 15   # 不确定天数
    infect_day = 10   # 感染者可传染的天数
    test_num = 30   # 测试集的样本量
    pred_day = window_forward + uncertain_day   # 预测天数

    country = 'India'
    inds = [30, 20, 10]

    dfs = [pd.read_csv('../result/disease_pre/India/10/' + country + '_' + str(ind) + '.csv') for ind in inds]
    sample_num = dfs[0].shape[0]   # the duration of COVID-19


    data_set = [drop_sample(df) for df in dfs]   # 去除早期样本


    # historical infection rates
    train_X_set = [df['infection_rate_10'][:-(ind + uncertain_day)].values.reshape(-1, 1) for ind, df in zip(inds, data_set)]
    # 指数函数预测pred_day后的infect_rate
    infect_rate_pred_set = [exp_func(list(range(len(train_X) + pred_day)), train_X).tolist() for train_X in train_X_set]


    # historical number of infection cases
    infect_num_set = [df['infection_delta_est'][:-(ind + uncertain_day)].values.tolist() for ind, df in zip(inds, dfs)]
    # calculate the number of infected cases
    infect_num_pred_set = [cal_infect_num(infect_num, infect_rate_pred[-pred_day:])
                  for infect_num, infect_rate_pred in zip(infect_num_set, infect_rate_pred_set)]


    # the real number of confirmed cases (one month)
    confirmed_delta_real = dfs[0]['confirmed_delta_real'][-30:].values.tolist()
    confirmed_delta_real = [confirmed_delta_real[i * 10:(i + 1) * 10] for i in range(3)]
    # calculate the number of confirmed cases using ISI model
    confirmed_delta_pred = [cal_confirmed_num(infect_num_pred) for infect_num_pred in infect_num_pred_set]
    # confirmed_delta_pred = [j for i in confirmed_delta_pred for j in i]

    #
    # # the error
    pred_rmse = np.mean([rmse(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(3)])
    pred_mape = np.mean([mape(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(3)])
    pred_mae = np.mean([mae(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(3)])
    #
    # # plot_confirmed(confirmed_delta_pred)


