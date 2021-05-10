import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter
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


def list_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c


def data_smooth(seq):
    seq_len = len(seq)
    seq_smooth = [0.7 * seq[0] + 0.3 * seq[1]]
    for i in range(1, seq_len-1):
        tmp = 0.3 * seq[i-1] + 0.4 * seq[i] + 0.3 * seq[i+1]
        seq_smooth.append(tmp)
    seq_smooth.append(0.7 * seq[-1] + 0.3 * seq[-2])

    return seq_smooth


def linear_fit(x, y, intercept):
    '''
    :param x: the number of days including fitted and pred
    :param infection_rate: infection rates needed to be fitted
    :return:
    '''
    lr = LinearRegression(fit_intercept=intercept)
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    lr.fit(x, y)
    param = lr.coef_[0]

    return param


def recip_fun(x):
    return np.sum((x[1] / (np.power(t_seq, x[2]) - x[0]) - beta_t) ** 2)
    # return (x[1] / (np.power(t_seq, x[2]) - x[0]) - y_true)


def jac_recip_func(x):
    return np.array([
        np.sum(2 * (x[1] / (np.power(t_seq, x[2]) - x[0]) - beta_t)
               * x[1] / (np.power(t_seq, x[2]) - x[0]) ** 2),
        np.sum(2 * (x[1] / (np.power(t_seq, x[2]) - x[0]) - beta_t)
               / (np.power(t_seq, x[2]) - x[0])),
        np.sum(2 * (x[1] / (np.power(t_seq, x[2]) - x[0]) - beta_t)
               * -x[1] / (np.power(t_seq, x[2]) - x[0]) ** 2 * x[2] * np.power(t_seq, x[2]-1))
    ])


def recip_fun_pred(t, param):
    a = param[0]
    b = param[1]
    c = param[2]
    return b / (np.power(t, c) - a)


def F(t, y, c):
    N = c
    dydt = [-recip_fun_pred(t, para.x) * y[1] * y[0] / N,
            recip_fun_pred(t, para.x) * y[1] * y[0] / N - gamma_est * y[1],
            gamma_est * y[1]
            ]
    return dydt


def plot_confirmed(test_pred):
    real = feature_country['confirmed_delta'].values.reshape(-1, 1)
    test_pred = np.array(test_pred).reshape(-1, 1)
    test_pred_merge = np.concatenate((real[-31].reshape(-1, 1), test_pred))

    plt.clf()
    plt.scatter(list(range(len(real))), real, s=5, c='r', label='Real Count')
    plt.plot(list(range((len(real)-(30+1)), len(real))), test_pred_merge,
             c='orange', label='vSIR prediction')
    plt.legend(loc='best')
    plt.title('Number of confirmed for ' + country + ' (' + str(window_forward) + ' days)')
    plt.vlines(len(real)-30, min(real), max(real), colors="grey", linestyles="dashed")
    plt.savefig('../result/figure/India/5/vSIR/confirmed_delta_' + country + '.png')


if __name__ == '__main__':
    os.chdir('../../')
    window_forward = 5   # 预测窗口
    population = 1.324 * 1e9


    country = 'India'
    disease = pd.read_csv('../data/disease_hist/' + country + '.csv', header=0)
    confirmed_delta_real = disease['confirmed_delta'][-30:].values.tolist()
    confirmed_delta_real = [confirmed_delta_real[i * 5:(i + 1) * 5] for i in range(6)]
    inds = [30, 25, 20, 15, 10, 5]


    confirmed_delta_pred = []
    for ind in inds:
        disease_copy = disease.iloc[34:-ind,:]
        confirmed_delta = disease_copy['confirmed_delta'].values.tolist()
        removed = list_add(disease_copy['death'].values.tolist(), disease_copy['recovered'].values.tolist())
        confirmed_delta_smooth =  data_smooth(confirmed_delta)
        removed_smooth =  data_smooth(removed)
        disease_smooth = pd.DataFrame({'date': disease_copy['date'],
                                       'confirmed_delta': confirmed_delta_smooth,
                                       'removed': removed_smooth})


        # plt.plot(list(range(len(confirmed_delta))), confirmed_delta)
        # plt.plot(list(range(len(confirmed_delta_smooth))), confirmed_delta_smooth)
        # plt.show()

        removed_diff = disease_smooth['removed'].diff(periods=2)
        disease_smooth['suseptible'] = population - disease_smooth['removed'] - disease_smooth['confirmed_delta']
        disease_smooth['removed_diff'] = removed_diff
        disease_smooth_drop = disease_smooth.dropna()

        gamma_est = linear_fit(disease_smooth_drop['confirmed_delta'], disease_smooth_drop['removed_diff'], False)
        tmp_t = []
        for i in range(5, disease_smooth_drop.shape[0]):
            tmp = linear_fit(list(range(i-5, i)), np.log(disease_smooth_drop['confirmed_delta'][i-5:i].values), True)
            tmp_t.append(tmp[0])
        beta_t = list_add(tmp_t, gamma_est.tolist() * len(tmp_t))
        t_seq = list(range(1, len(tmp_t)+1))


        bounds = ([-np.inf, 0, 0], [1, np.inf, np.inf])
        param0 = [-0.01, 1, 1]
        # para = least_squares(recip_fun, param0, jac_recip_func, bounds=bounds) # 进行拟合
        para = least_squares(recip_fun, param0, bounds=bounds) # 进行拟合

        # y_fitted = recip_fun_pred(t_seq, para.x)
        # plt.plot(t_seq, beta_t, 'r', label='Original curve')
        # plt.plot(t_seq, y_fitted, '-b', label='Fitted curve')
        # plt.legend()
        # plt.show()

        tspan = np.linspace(len(tmp_t), len(tmp_t)+window_forward, 6)
        yinit = [disease_smooth_drop['suseptible'][len(tmp_t)],
                 disease_smooth_drop['confirmed_delta'][len(tmp_t)],
                 disease_smooth_drop['removed'][len(tmp_t)]]
        sol = solve_ivp(lambda t, y: F(t, y, population),
                  [tspan[0], tspan[-1]], yinit,
                  t_eval=tspan, rtol=1e-5)
        # state_plotter(sol.t, sol.y, 1)
        confirmed_delta_pred.append(sol.y[1,1:].tolist())


    pred_rmse = np.mean([rmse(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(6)])
    pred_mape = np.mean([mape(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(6)])
    pred_mae = np.mean([mae(confirmed_delta_real[i], confirmed_delta_pred[i]) for i in range(6)])


    # model_res(test_pred_0, ind)
    with open('../result/figure/India/5/vSIR/' + country + '_model_res.txt', 'a') as file:
        file.write('test_rmse: {:d}, test_mae: {:d}, test_mape: {:.6f}'.format(
            round(pred_rmse), round(pred_mae), pred_mape))
        file.write('\n')









