import pandas as pd
import numpy as np
import torch
import datetime
from sklearn import preprocessing
import os
import random


def disease():
    '''
    prepare the diseased related data (except normalization)
    :return: diseased related data (except normalization)
    '''
    # raw historical confirmed cases
    confirmed = pd.read_csv('../data/time_series_covid19_confirmed_global.csv', header=0)
    # raw historical dead cases
    death = pd.read_csv('../data/time_series_covid19_deaths_global.csv', header=0)
    # raw historical recovered cases
    recovered = pd.read_csv('../data/time_series_covid19_recovered_global.csv', header=0)
    # name of countries or regions
    region_name = sorted(list(set(confirmed['Country/Region'])))

    disease_info = {}.fromkeys(region_name)
    confirmed.drop(columns=['Lat', 'Long'], inplace=True)
    death.drop(columns=['Lat', 'Long'], inplace=True)
    recovered.drop(columns=['Lat', 'Long'], inplace=True)

    # format the date to '%Y-%m-%d'
    date = confirmed.columns[2:]
    date = [datetime.date(2020, int(d.split('/')[0]), int(d.split('/')[1])) for d in date]
    date = [d.__format__('%Y-%m-%d') for d in date]

    for name in region_name:
        # aggregate the confirmed cases in the same countries or regions
        region_confirmed = confirmed[confirmed['Country/Region'] == name].iloc[:, 2:].agg(np.sum)
        # aggregate the dead cases in the same countries or regions
        region_death = death[death['Country/Region'] == name].iloc[:, 2:].agg(np.sum)
        # aggregate the recovered cases in the same countries or regions
        region_recovered = recovered[recovered['Country/Region'] == name].iloc[:, 2:].agg(np.sum)

        # new daily confirmed cases
        region_confirmed_delta = [region_confirmed[0]]
        region_confirmed_delta.extend(np.diff(region_confirmed, n=1).tolist())

        # new daily dead cases
        region_death_delta = [region_death[0]]
        region_death_delta.extend(np.diff(region_death, n=1).tolist())

        # new daily recovered cases
        region_recovered_delta = [region_recovered[0]]
        region_recovered_delta.extend(np.diff(region_recovered, n=1).tolist())

        # current number of hospitalizations
        region_hospital = region_confirmed - region_recovered - region_death

        temp = pd.DataFrame({'date': date, 'region': [name] * len(date),
                             'confirmed': region_confirmed.values.tolist(),
                             'death': region_death.values.tolist(),
                             'recovered': region_recovered.values.tolist(),
                             'confirmed_delta': region_confirmed_delta,
                             'death_delta': region_death_delta,
                             'recovered_delta': region_recovered_delta,
                             'hospital': region_hospital.tolist()})
        temp = temp[(temp.iloc[:, 2] != 0)]  # 去除疫情没有开始的日期
        temp.reset_index(inplace=True, drop=True)
        disease_info[name] = temp
        if name == 'Taiwan*':
            continue
        temp.to_csv('../data/disease_hist/' + name + '.csv', index=False)

    return disease_info


def mobility(region_name):
    '''
    prepare the mobility data (except normalization)
    :param region_name: name of region
    :return: mobility info of region
    '''
    # raw historical mobility data
    mobility_global = pd.read_csv('../data/Global_Mobility_Report.csv', header=0, low_memory=False)
    region = [region_name]
    mobility_info = {}.fromkeys(region)

    for name in region:
        temp = mobility_global[mobility_global['country_region'] == name]
        temp = temp[(pd.isnull(temp['sub_region_1'])) & (pd.isnull(temp['sub_region_2'])) & (pd.isnull(temp['metro_area']))]
        mobi_type = temp.columns[8:14]   # six types of mobility
        col = ['country_region', 'date']
        col.extend(mobi_type)
        temp = temp[col]
        mobi_type = [i.split('_percent_change_from_baseline')[0] for i in temp.columns[2:8]]   # names of six types of mobility
        col = ['country_region', 'date']
        col.extend(mobi_type)
        temp.columns = col   # update the colname of temp
        temp.reset_index(inplace=True, drop=True)
        temp.drop('country_region', axis=1, inplace=True)
        mobility_info[name] = temp

    return mobility_info


def drop_sample(data):
    '''
    如果day t的新增确诊数量为0，且day t+1 and day t+2的新增确诊数量也为0，则去除day t的样本
    :param data: raw COVID-19 daily dataset released by Johns Hopkins University.
    :return: preprocessed data.
    '''
    confirmed_delta = data['confirmed_delta']
    ind = 0
    for i in range(len(confirmed_delta)-2):
        if ((confirmed_delta[i] != 0) & (confirmed_delta[i+1] != 0) & (confirmed_delta[i+2] != 0)):
            ind = i
            break
    data = data.iloc[ind:,:]

    return data, ind


def merge_hist(mobility_info, disease_info, country_name, mask_ind):
    '''
    merge the disease-related data and mobility info
    :param mobility_info: the six type of mobility
    :param disease_info: disease-related data
    :param country_name: country name
    :param mask_ind:
    :return: prepared features, the index of the dropped row
    '''
    if mask_ind == None:
        disease_pre = pd.read_csv('../result/disease_pre/India/10/' + country_name + '.csv', header=0)
    else:
        disease_pre = pd.read_csv('../result/disease_pre/India/10/' + country_name + '_' + str(mask_ind) + '.csv', header=0)
    disease_region = disease_info[country_name]
    disease_region = disease_region.drop(['confirmed', 'death', 'recovered'], axis=1)
    disease_region, ind = drop_sample(disease_region)
    temp = pd.merge(disease_region, mobility_info[country_name], how='inner', on='date')
    temp = pd.merge(temp, disease_pre, how='left', on='date')
    temp.drop(['confirmed_delta_real', 'confirmed_delta_est', 'infection_delta_est'], axis=1, inplace=True)
    temp_selected = temp.iloc[:,:-14]   # 去掉感染率列
    temp_selected['infect_rate'] = temp['infection_rate_10']
    if mask_ind == None:
        return temp_selected, ind
    else:
        temp_selected.to_csv('../result/feature_merge/' + country_name + '_' + str(mask_ind) + '.csv', index=False)
        return temp_selected


def maxmin_scale(raw_data):
    '''
    0-1 normalize the data
    :param raw_data: raw data
    :return: normalized data
    '''
    label = raw_data.iloc[:, -1].values.reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=[0,1])
    label_scaled = scaler.fit_transform(label)
    raw_data = raw_data.iloc[:, 2:].values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=[0,1])
    disease_scaled = min_max_scaler.fit_transform(raw_data)
    disease_scaled = pd.DataFrame(disease_scaled)

    return disease_scaled, label_scaled, scaler, min_max_scaler


def hybrid_scale(raw_data):
    '''
    hybrid normalization (infection rate:z_score; else feature:0-1 normalization)
    :param raw_data: raw data
    :return: normalized data
    '''
    label = raw_data.iloc[:, -1].values.reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    label_scaled = scaler.fit_transform(label)
    raw_data = raw_data.iloc[:, 2:-1].values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=[0,1])
    disease_scaled = min_max_scaler.fit_transform(raw_data)
    # feature_merge = np.concatenate((disease_scaled, label_scaled), axis=1)
    # feature_merge = pd.DataFrame(feature_merge)
    feature_merge = pd.DataFrame(disease_scaled)

    return feature_merge, label_scaled, scaler, min_max_scaler


def z_score_scale(raw_data):
    '''
    z_score normalize the data
    :param raw_data: raw data
    :return: normalized data
    '''
    label = raw_data.iloc[:, -1].values.reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    label_scaled = scaler.fit_transform(label)
    raw_data = raw_data.iloc[:, 2:].values
    z_score_scaler = preprocessing.StandardScaler()
    disease_scaled = z_score_scaler.fit_transform(raw_data)
    disease_scaled = pd.DataFrame(disease_scaled)

    return disease_scaled, label_scaled, scaler, z_score_scaler


def series_to_supervised(data_copy, seq_len):
    data = data_copy.copy()
    data.loc['new_row'] = 1

    # if the number of feature is one!
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # input sequence (t-seq_len, ... t-1)
    for i in range(seq_len, 0, -1):
        # down shift i days
        cols.append(df.shift(i))
        # the colname corresponding the the shifted column
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    return agg


def data_split(train_val_data, label, window_backward, pred_day, n_features):
    '''
    :param train_val_data:
    :param label:
    :param window_backward:
    :param pred_day:
    :param n_features:
    :return:
    '''
    train_val_y = [label[i: i + pred_day].tolist() for i in range(window_backward, len(label) - pred_day + 1)]
    train_val_num = len(train_val_y)

    train_val_X = train_val_data.values[:train_val_num,:]
    train_num = int(train_val_num * 0.9)   # number of validation set is 5

    # randomly choose train_ind and val_ind
    train_ind = sorted(random.sample(range(train_val_num), train_num))
    val_ind = [i for i in range(train_val_num) if i not in train_ind]
    train_X = np.array([train_val_X[i,:].tolist() for i in train_ind])
    train_y = np.array([train_val_y[i] for i in train_ind])
    val_X = np.array([train_val_X[i,:].tolist() for i in val_ind])
    val_y = np.array([train_val_y[i] for i in val_ind])

    train_X = torch.tensor(train_X, dtype=torch.float).reshape(-1, window_backward, n_features)
    train_y = torch.tensor(train_y, dtype=torch.float).reshape(-1, pred_day)
    val_X = torch.tensor(val_X, dtype=torch.float).reshape(-1, window_backward, n_features)
    val_y = torch.tensor(val_y, dtype=torch.float).reshape(-1, pred_day)
    train_val_X = torch.tensor(train_val_X, dtype=torch.float).reshape(-1, window_backward, n_features)
    train_val_y = torch.tensor(train_val_y, dtype=torch.float).reshape(-1, pred_day)

    return train_X, train_y, val_X, val_y, train_val_X, train_val_y


if __name__ == '__main__':
    os.chdir('../')
    seed = 100
    np.random.seed(seed)
    random.seed(seed)


    window_backward = 10
    window_forward = 3
    uncertain_day = 15
    n_features = 11
    pred_day = uncertain_day + window_forward
    test_num = 30 + uncertain_day
    country = 'India'


    disease = disease()
    mobility = mobility(country)
    feature_country, ind = merge_hist(mobility, disease, country, None)

    train_val_data = feature_country.iloc[:-test_num,:]
    train_val_data_scaled, label = train_val_data.iloc[:,2:], train_val_data.iloc[:,-1].values
    train_val_data_scaled_sample = series_to_supervised(train_val_data_scaled, window_backward)


    # data_scaled, label, scaler, min_max_scaler = maxmin_scale(feature_country)
    train_X, train_y, val_X, val_y, train_val_X, train_val_y = data_split(
        train_val_data_scaled_sample, label, window_backward, pred_day, n_features)
