from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import where, searchsorted
from pyod.models.knn import KNN
from scipy.stats import entropy, norm
from sklearn import preprocessing, tree, metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def preprocess(dataset, columns_to_scale, columns_to_encode, label, columns_to_drop):
    df = pd.read_csv(dataset)
    df = df.drop(columns_to_drop, axis=1)
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(df[columns_to_encode])
    df_encoded = pd.concat([df.drop(columns_to_encode, 1), pd.DataFrame(ohe.transform(df[columns_to_encode]))],
                           axis=1).reindex()

    clf = LocalOutlierFactor(contamination=0.1)
    outlier_index = where(clf.fit_predict(df_encoded) == -1)
    no_outlier_df = df_encoded.drop(outlier_index[0], axis=0)
    y = no_outlier_df[label].to_numpy()
    x = no_outlier_df.drop(label, axis=1)

    outlier_df = df_encoded.drop(x.index)
    y_outlier = outlier_df[label]
    x_outlier = outlier_df.drop(label, axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_columns = min_max_scaler.fit_transform(x[columns_to_scale])
    encoded_columns = x.drop(columns_to_scale, axis=1).to_numpy()
    x = np.concatenate([scaled_columns, encoded_columns], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.80, random_state=1)

    outlier_scaled_columns = min_max_scaler.transform(x_outlier[columns_to_scale])
    x_outlier = np.concatenate([outlier_scaled_columns, x_outlier.drop(columns_to_scale, axis=1).to_numpy()], axis=1)
    x_test = np.concatenate([x_test, x_outlier])
    y_test = np.concatenate([y_test, y_outlier])

    return x_train, y_train, x_test, y_test


def preprocess_3d_road_network(iteration=0):
    df = pd.read_csv("data/3d_road/" + str(iteration) + ".csv")
    x = df.drop('alt', axis=1)
    y_train = df.alt
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    df = pd.read_csv("data/3d_road/test_3d_road_network.csv")
    x_test = df.drop('alt', axis=1)
    y_test = df.alt
    x_test = min_max_scaler.fit_transform(x_test)

    return x_train, y_train, x_test, y_test


def calculate_distrust_values(x_train, y_train, x_test, y_test, num_of_neighbors, contamination_rate):
    # metrics = ['braycurtis', 'canberra', 'chebyshev', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
    clf = KNN(n_neighbors=num_of_neighbors, metric='minkowski')
    clf.fit(x_train)
    distance_test, neighbours_test = clf.get_neighbours(x_test)
    y_test_scores = clf.decision_function(x_test)
    y_train_scores = clf.decision_scores_

    wdt_scores_list = []
    sdt_scores_list = []
    uncertainty_list = []
    outlierness_score = []

    train_uncertainty_score = []
    distance_train, neighbours_train = clf.get_neighbours(x_train)
    for idx, val in enumerate(x_train):
        neighborhood = [y_train[i] for i in neighbours_train[idx]]
        avg = np.ones(num_of_neighbors) * np.mean(neighborhood)
        train_uncertainty_score.append(10 * mean_squared_error(neighborhood, avg))

    mean_uncertainty = 1.0 - contamination_rate
    sd_uncertainty = 0.1
    train_uncertainty_score.sort()
    uncertainty_mean_index = int(mean_uncertainty * len(train_uncertainty_score))

    for idx, val in enumerate(x_test):
        neighborhood = [y_train[i] for i in neighbours_test[idx]]
        avg = np.ones(num_of_neighbors) * np.mean(neighborhood)
        error = 10 * mean_squared_error(neighborhood, avg)
        query_index_ = searchsorted(train_uncertainty_score, error, side='left', sorter=None)
        percentile_ = (mean_uncertainty * query_index_) / uncertainty_mean_index
        z_score_ = (percentile_ - mean_uncertainty) / sd_uncertainty
        uncertainty_list.append(norm.cdf(z_score_))

    mean_outlierness = 1.0 - contamination_rate
    sd_outlierness = 0.1
    y_train_scores.sort()
    outlier_mean_index = int(mean_outlierness * len(y_train_scores))

    for idx, val in enumerate(y_test_scores):
        query_index = searchsorted(y_train_scores, val, side='left', sorter=None)
        percentile = (mean_outlierness * query_index) / outlier_mean_index
        z_score = (percentile - mean_outlierness) / sd_outlierness
        outlierness_score.append(norm.cdf(z_score))

    wdt_x_dict = defaultdict(list)
    sdt_x_dict = defaultdict(list)
    wdt_y_dict = defaultdict(list)
    sdt_y_dict = defaultdict(list)

    for i in range(len(x_test)):
        sdt_scores_list.append(
            uncertainty_list[i] + outlierness_score[i] - (uncertainty_list[i] * outlierness_score[i]))
        wdt_scores_list.append(uncertainty_list[i] * outlierness_score[i])

        if int(np.ceil(10 * uncertainty_list[i] * uncertainty_list[i])) != 0:
            wdt_x_dict[int(np.ceil(10 * uncertainty_list[i] * uncertainty_list[i]))].append(x_test[i])
            wdt_y_dict[int(np.ceil(10 * uncertainty_list[i] * uncertainty_list[i]))].append(y_test[i])
        else:
            wdt_x_dict[int(np.ceil(10 * uncertainty_list[i] * uncertainty_list[i])) + 1].append(x_test[i])
            wdt_y_dict[int(np.ceil(10 * uncertainty_list[i] * uncertainty_list[i])) + 1].append(y_test[i])

        sdt_x_dict[int(np.ceil(
            10 * (uncertainty_list[i] + uncertainty_list[i] - (uncertainty_list[i] * uncertainty_list[i]))))].append(
            x_test[i])
        sdt_y_dict[int(np.ceil(
            10 * (uncertainty_list[i] + uncertainty_list[i] - (uncertainty_list[i] * uncertainty_list[i]))))].append(
            y_test[i])

    return wdt_x_dict, wdt_y_dict, sdt_x_dict, sdt_y_dict


def calculate_stats(model, distrust_x_dict, distrust_y_dict):
    mse_list = []
    bucket_size = []
    for key in sorted(distrust_x_dict):
        bucket_size.append(len(distrust_x_dict[key]))
        y_pred = model.predict(distrust_x_dict[key])
        y_true = distrust_y_dict[key]
        mse_list.append(metrics.mean_squared_error(y_true, y_pred))
    return mse_list


def effectiveness_exp(mse_list, measure):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8',
                    '0.8-0.9', '0.9-1.0']
    x_axis_tick = np.arange(len(mse_list))

    ax.bar(x_axis_tick, mse_list, width=0.5, color='b', label='RSS')
    ax.set_xticks(x_axis_tick)
    ax.legend()
    ax.set_xticklabels(x_axis_label[:len(mse_list)], rotation=45)
    ax.set_yticks(y_axis_tick)
    ax.title.set_text("Effectiveness of " + measure)
    ax.set_xlabel(measure)
    plt.show()


def run_effectiveness_exp(dataset, columns_to_scale, columns_to_encode, label, columns_to_drop):
    num_of_neighbors = 50
    contamination_rate = 0.1
    x_train, y_train, x_test, y_test = preprocess(dataset=dataset,
                                                  columns_to_scale=columns_to_scale,
                                                  columns_to_encode=columns_to_encode, label=label,
                                                  columns_to_drop=columns_to_drop)
    wdt_x_dict, wdt_y_dict, sdt_x_dict, sdt_y_dict = calculate_distrust_values(x_train, y_train, x_test, y_test,
                                                                               num_of_neighbors, contamination_rate)
    # model = Sequential()
    # model.add(Dense(128, input_shape=(26,), kernel_initializer='normal', activation='relu'))
    # model.add(Dense(units=64, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(32, input_shape=(26,), kernel_initializer='normal', activation='relu'))
    # model.add(Dense(units=16, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=150, batch_size=10)
    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)

    mse_list_sdt = calculate_stats(model, sdt_x_dict, sdt_y_dict)
    mse_list_wdt = calculate_stats(model, wdt_x_dict, wdt_y_dict)

    effectiveness_exp(mse_list_sdt, measure='SDT')
    effectiveness_exp(mse_list_wdt, measure='WDT')


def run_3d_road_network_effectiveness_exp():
    num_of_neighbors = 50
    contamination_rate = 0.1

    x_train, y_train, x_test, y_test = preprocess_3d_road_network()
    wdt_x_dict, wdt_y_dict, sdt_x_dict, sdt_y_dict = calculate_distrust_values(x_train, y_train, x_test, y_test,
                                                                               num_of_neighbors, contamination_rate)
    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)

    mse_list_sdt = calculate_stats(model, sdt_x_dict, sdt_y_dict)
    mse_list_wdt = calculate_stats(model, wdt_x_dict, wdt_y_dict)

    effectiveness_exp(mse_list_sdt, measure='SDT')
    effectiveness_exp(mse_list_wdt, measure='WDT')


# run_effectiveness_exp(dataset="data/kc_house_data.csv",
#                       columns_to_scale=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
#                                         'waterfront', 'view', 'condition', 'grade',
#                                         'sqft_above', 'sqft_basement', 'yr_built',
#                                         'yr_renovated', 'lat', 'long', 'sqft_living15',
#                                         'sqft_lot15'],
#                       columns_to_encode=['floors', 'zipcode'], label='price',
#                       columns_to_drop=['id', 'date'])
#
# run_effectiveness_exp(dataset="data/diamonds.csv",
#                       columns_to_scale=['carat', 'depth', 'table', 'x', 'y', 'z'],
#                       columns_to_encode=['cut', 'color', 'clarity'], label='price',
#                       columns_to_drop=['id'])

run_3d_road_network_effectiveness_exp()
