import math
import random
from collections import defaultdict
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from numpy import where, searchsorted
from pyod.models.knn import KNN
from scipy.stats import entropy, norm
from sdv.evaluation import evaluate
from sdv.tabular import CTGAN, GaussianCopula, CopulaGAN, TVAE
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

min_max_scaler = preprocessing.MinMaxScaler()
ohe = OneHotEncoder(sparse=False)
pd.options.display.max_seq_items = None


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.3f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


def shannon_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def preprocess(dataset, columns_to_scale, columns_to_encode, label, columns_to_drop):
    df = pd.read_csv(dataset)
    df = df.drop(columns_to_drop, axis=1)
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

    scaled_columns = min_max_scaler.fit_transform(x[columns_to_scale])
    encoded_columns = x.drop(columns_to_scale, axis=1).to_numpy()
    x = np.concatenate([scaled_columns, encoded_columns], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.80, random_state=1)

    outlier_scaled_columns = min_max_scaler.transform(x_outlier[columns_to_scale])
    x_outlier = np.concatenate([outlier_scaled_columns, x_outlier.drop(columns_to_scale, axis=1).to_numpy()], axis=1)
    x_test = np.concatenate([x_test, x_outlier])
    y_test = np.concatenate([y_test, y_outlier])

    return x_train, y_train, x_test, y_test


def preprocess_synthetic(iteration=1):
    df = pd.read_csv("data/synthetic/train/" + str(iteration) + ".csv")
    x_train = df.drop('Y', axis=1).to_numpy()
    y_train = df.Y.to_numpy()
    df = pd.read_csv("data/synthetic/test/" + str(iteration) + ".csv")
    x_test = df.drop('Y', axis=1).to_numpy()
    y_test = df.Y.to_numpy()
    return x_train, y_train, x_test, y_test


def generate_synthetic_sample(n, dataset, columns_to_scale, columns_to_encode, label):
    df = pd.read_csv(dataset)
    df = df.drop(label, axis=1)
    columns = columns_to_scale + columns_to_encode
    model = GaussianCopula()
    model.fit(df)
    synthetic_sample = model.sample(n)
    print("synthetic data evaluation:", evaluate(synthetic_sample, df))
    df = pd.DataFrame(synthetic_sample, columns=columns)
    df = pd.concat([df.drop(columns_to_scale, 1),
                    pd.DataFrame(min_max_scaler.transform(df[columns_to_scale]), columns=columns_to_scale)],
                   axis=1).reindex()
    df = pd.concat([df.drop(columns_to_encode, 1), pd.DataFrame(ohe.transform(df[columns_to_encode]))],
                   axis=1).reindex()
    return df.values.tolist()


def generate_random_sample(n, dataset, columns_to_scale, columns_to_encode):
    df = pd.read_csv(dataset)
    columns = columns_to_scale + columns_to_encode

    random_sample = []
    for i in range(n):
        random_point = []
        for idx, value in enumerate(columns):
            if value in columns_to_scale:
                random_point.append(random.uniform(0, 1))
            else:
                column_values = df[value].unique().tolist()
                random_point.append(random.choice(column_values))
        random_sample.append(random_point)

    df = pd.DataFrame(random_sample, columns=columns)
    df = pd.concat([df.drop(columns_to_encode, 1), pd.DataFrame(ohe.transform(df[columns_to_encode]))],
                   axis=1).reindex()
    return df.values.tolist()


def calculate_distrust_values(x_train, y_train, x_test, y_test, num_of_neighbors, contamination_rate, n_samples,
                              dataset,
                              columns_to_scale,
                              columns_to_encode, label):
    # metrics = ['braycurtis', 'canberra', 'chebyshev', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching',
    #            'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
    clf = KNN(n_neighbors=num_of_neighbors, metric='minkowski')
    clf.fit(x_train)
    # distance, neighbours = clf.get_neighbours(x_test)
    train_outlier_score = clf.decision_scores_
    # test_outlier_score = clf.decision_function(x_test)
    wdt_scores_list = []
    sdt_scores_list = []

    mean = 1.0 - contamination_rate
    sd = 0.1
    train_outlier_score.sort()
    outlier_mean_index = int(mean * len(train_outlier_score))

    wdt_x_dict = defaultdict(list)
    sdt_x_dict = defaultdict(list)
    wdt_y_dict = defaultdict(list)
    sdt_y_dict = defaultdict(list)

    # _________________________________learning entropy_____________________________________
    print("learning entropy")
    n = n_samples
    epsilon = 0.005
    param_grid = [{'bootstrap': [False, True], 'n_estimators': [100, 200, 500, 1000, 2000],
                   'max_features': [1.0, 'sqrt', 'log2']}]
    model_entropy = RandomForestRegressor()
    grid_search_entropy = GridSearchCV(model_entropy, param_grid, cv=5, scoring='neg_mean_squared_error',
                                       return_train_score=True)

    while True:
        x_random_sample = generate_random_sample(n=int(n / 10), dataset=dataset, columns_to_scale=columns_to_scale,
                                                 columns_to_encode=columns_to_encode)
        x_synthetic_sample = generate_synthetic_sample(n=n, dataset=dataset, columns_to_scale=columns_to_scale,
                                                       columns_to_encode=columns_to_encode, label=label)
        x_sample = x_random_sample + x_synthetic_sample
        _, neighbours_sample = clf.get_neighbours(x_sample)
        y_sample = []

        for idx in range(len(x_sample)):
            labels = [y_train[i] for i in neighbours_sample[idx]]
            y_sample.append(shannon_entropy(labels))

        x_sample_train, x_sample_test, y_sample_train, y_sample_test = train_test_split(
            x_sample, y_sample, test_size=0.80, random_state=1)
        grid_search_entropy.fit(x_sample_train, y_sample_train)
        print("best model: ", grid_search_entropy.best_params_)
        mse_test = metrics.mean_squared_error(y_sample_test, grid_search_entropy.predict(x_sample_test), squared=False)
        print("test uncertainty mse:", mse_test)
        if mse_test < epsilon:
            break
        else:
            n *= 2
            print("number of samples", n)
    # ___________________________________learning outlier distance__________________________________________
    n = n_samples
    print("learning outlier distance")
    epsilon = 0.001
    model_outlier_distance = RandomForestRegressor()
    grid_search_outlier = GridSearchCV(model_outlier_distance, param_grid, cv=5, scoring='neg_mean_squared_error',
                                       return_train_score=True)

    while True:
        x_random_sample = generate_random_sample(n=int(n / 2), dataset=dataset, columns_to_scale=columns_to_scale,
                                                 columns_to_encode=columns_to_encode)
        x_synthetic_sample = generate_synthetic_sample(n=int(n / 2), dataset=dataset, columns_to_scale=columns_to_scale,
                                                       columns_to_encode=columns_to_encode, label=label)
        x_sample = x_random_sample + x_synthetic_sample
        y_sample = clf.decision_function(x_sample)

        x_sample_train, x_sample_test, y_sample_train, y_sample_test = train_test_split(
            x_sample, y_sample, test_size=0.80, random_state=1)
        grid_search_outlier.fit(x_sample_train, y_sample_train)
        print("best model: ", grid_search_outlier.best_params_)

        mse_test = metrics.mean_squared_error(y_sample_test, grid_search_outlier.predict(x_sample_test),
                                              squared=False)
        print("test outlier mse:", mse_test)
        if mse_test < epsilon:
            break
        else:
            n *= 2
            print("number of samples", n)
    # _______________________________________Real values_____________________________________________
    # for idx, val in enumerate(test_outlier_score):
    #     labels = [y_train[i] for i in neighbours[idx]]
    #     uncertainty = shannon_entropy(labels)
    #     query_index = searchsorted(train_outlier_score, val, side='left', sorter=None)
    #     percentile = (mean * query_index) / outlier_mean_index
    #     z_score = (percentile - mean) / sd
    #     outlierness = norm.cdf(z_score)

    # _______________________________________Predicted values_____________________________________________
    for idx in range(len(x_test)):
        uncertainty = grid_search_entropy.predict(x_test[idx].reshape(1, -1))[0]
        query_index = searchsorted(train_outlier_score, grid_search_outlier.predict(x_test[idx].reshape(1, -1))[0],
                                   side='left', sorter=None)
        percentile = (mean * query_index) / outlier_mean_index
        z_score = (percentile - mean) / sd
        outlierness = norm.cdf(z_score)
        sdt_scores_list.append(uncertainty + outlierness - (uncertainty * outlierness))
        wdt_scores_list.append(uncertainty * outlierness)

        if int(np.ceil(10 * uncertainty * outlierness)) != 0:
            wdt_x_dict[int(np.ceil(10 * uncertainty * outlierness))].append(x_test[idx])
            wdt_y_dict[int(np.ceil(10 * uncertainty * outlierness))].append(y_test[idx])
        else:
            wdt_x_dict[int(np.ceil(10 * uncertainty * outlierness)) + 1].append(x_test[idx])
            wdt_y_dict[int(np.ceil(10 * uncertainty * outlierness)) + 1].append(y_test[idx])

        sdt_x_dict[int(np.ceil(10 * (uncertainty + outlierness - (uncertainty * outlierness))))].append(x_test[idx])
        sdt_y_dict[int(np.ceil(10 * (uncertainty + outlierness - (uncertainty * outlierness))))].append(y_test[idx])

    return wdt_x_dict, wdt_y_dict, sdt_x_dict, sdt_y_dict, sdt_scores_list, wdt_scores_list


def calculate_stats(model, distrust_x_dict, distrust_y_dict):
    accuracy_list = []
    f1_list = []
    fnr_list = []
    fpr_list = []
    bucket_size = []

    for key in sorted(distrust_x_dict):
        bucket_size.append(len(distrust_x_dict[key]))
        y_pred = model.predict(distrust_x_dict[key])
        y_true = distrust_y_dict[key]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        f1 = tp / (tp + 0.5 * (fp + fn))
        acc = (tp + tn) / (tp + fp + fn + tn)

        if fn == 0:
            fnr = 0
        else:
            fnr = fn / (fn + tp)

        if fp == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        accuracy_list.append(acc)
        fnr_list.append(fnr)
        fpr_list.append(fpr)
        f1_list.append(f1)
    return accuracy_list, f1_list, fnr_list, fpr_list


def effectiveness_exp(accuracy_list, f1_list, fnr_list, fpr_list, measure):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    y_axis_tick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    x_axis_label = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8',
                    '0.8-0.9', '0.9-1.0']
    x_axis_tick = np.arange(len(accuracy_list))

    ax.bar(x_axis_tick - 0.3, accuracy_list, color='b', width=0.2, label='Accuracy')
    ax.bar(x_axis_tick - 0.1, f1_list, color='g', width=0.2, label='F1')
    ax.bar(x_axis_tick + 0.1, fnr_list, color='r', width=0.2, label='FNR')
    ax.bar(x_axis_tick + 0.3, fpr_list, color='y', width=0.2, label='FPR')
    ax.legend()
    ax.set_xticks(x_axis_tick)
    ax.set_xticklabels(x_axis_label[:len(accuracy_list)], rotation=45)
    ax.set_yticks(y_axis_tick)
    ax.title.set_text("Effectiveness of " + measure)
    ax.set_xlabel(measure)
    plt.savefig("results/" + measure + ".png")


def run_effectiveness_exp(dataset, columns_to_scale, columns_to_encode, label, columns_to_drop=[]):
    num_of_neighbors = 50
    contamination_rate = 0.1
    x_train, y_train, x_test, y_test = preprocess(dataset=dataset,
                                                  columns_to_scale=columns_to_scale,
                                                  columns_to_encode=columns_to_encode, label=label,
                                                  columns_to_drop=columns_to_drop)
    wdt_x_dict, wdt_y_dict, sdt_x_dict, sdt_y_dict, _, _ = calculate_distrust_values(x_train, y_train, x_test, y_test,
                                                                                     num_of_neighbors,
                                                                                     contamination_rate,
                                                                                     n_samples=1000,
                                                                                     dataset=dataset,
                                                                                     columns_to_scale=columns_to_scale,
                                                                                     columns_to_encode=columns_to_encode,
                                                                                     label=label)

    # DL Model
    # model = Sequential()
    # model.add(Dense(5, input_shape=(4,), activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(x, y, epochs=150, batch_size=10)
    model = MLPClassifier(max_iter=1000)
    model.fit(x_train, y_train)

    accuracy_list_sdt, f1_list_sdt, fnr_list_sdt, fpr_list_sdt = calculate_stats(model, sdt_x_dict, sdt_y_dict)
    accuracy_list_wdt, f1_list_wdt, fnr_list_wdt, fpr_list_wdt = calculate_stats(model, wdt_x_dict, wdt_y_dict)

    effectiveness_exp(accuracy_list_sdt, f1_list_sdt, fnr_list_sdt, fpr_list_sdt, measure='SDT')
    effectiveness_exp(accuracy_list_wdt, f1_list_wdt, fnr_list_wdt, fpr_list_wdt, measure='WDT')


def run_synthetic_effectiveness_exp():
    num_of_neighbors = 50
    contamination_rate = 0.1
    iteration = 1

    x_train, y_train, x_test, y_test = preprocess_synthetic()
    wdt_x_dict, wdt_y_dict, sdt_x_dict, sdt_y_dict, sdt_scores_list, wdt_scores_list = calculate_distrust_values(
        x_train, y_train, x_test, y_test,
        num_of_neighbors, contamination_rate, n_samples=1000, dataset="data/synthetic/train/" + str(iteration) + ".csv",
        columns_to_scale=['X_1', 'X_2'],
        columns_to_encode=[], label='Y')
    model = MLPClassifier(max_iter=2000)
    model.fit(x_train, y_train)

    accuracy_list_sdt, f1_list_sdt, fnr_list_sdt, fpr_list_sdt = calculate_stats(model, sdt_x_dict, sdt_y_dict)
    accuracy_list_wdt, f1_list_wdt, fnr_list_wdt, fpr_list_wdt = calculate_stats(model, wdt_x_dict, wdt_y_dict)

    effectiveness_exp(accuracy_list_sdt, f1_list_sdt, fnr_list_sdt, fpr_list_sdt, measure='SDT')
    effectiveness_exp(accuracy_list_wdt, f1_list_wdt, fnr_list_wdt, fpr_list_wdt, measure='WDT')

    distrust_measure = ["SDT", "WDT"]
    for idx, value in enumerate([sdt_scores_list, wdt_scores_list]):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        minima = np.min(value)
        maxima = np.max(value)
        normalized = Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=normalized, cmap=cm.RdYlGn_r)
        color = []
        for v in value:
            color.append(mapper.to_rgba(v))
        ax.scatter([item[0] for item in x_test], [item[1] for item in x_test], color=[item for item in color], s=2)
        ax.set_aspect('equal', adjustable='box')
        ax.title.set_text(distrust_measure[idx])
        plt.savefig("results/" + distrust_measure[idx] + "_visualization.png")

    step = 0.1
    grid_feature = {}
    grid_label = {}
    index = 0
    for i in np.arange(0.0, 1.0, step):
        for j in np.arange(0.0, 1.0, step):
            features = []
            labels = []
            for k in range(len(x_test)):
                if i <= x_test[k][0] <= i + step and j <= x_test[k][1] <= j + step:
                    features.append(x_test[k])
                    labels.append(y_test[k])
                grid_feature.update({index: features})
                grid_label.update({index: labels})
            index += 1

    accuracy_list = []
    f1_list = []
    fpr_list = []
    fnr_list = []

    for i in range(index):
        y_pred = model.predict(grid_feature[i])
        y_true = grid_label[i]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = (tp + tn) / (tp + fp + fn + tn)

        if tp == 0:
            f1 = 0
        else:
            f1 = tp / (tp + 0.5 * (fp + fn))

        if fp == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        if fn == 0:
            fnr = 0
        else:
            fnr = fn / (fn + tp)

        accuracy_list.append(acc)
        f1_list.append(f1)
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    x_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    measure = ["Accuracy", "F1", "FPR", "FNR"]
    measure_values = [accuracy_list, f1_list, fpr_list, fnr_list]

    for idx, value in enumerate(measure_values):
        B = np.reshape(value, (len(y_axis), len(x_axis)))
        measure_list = B.T
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
        im, cbar = heatmap(measure_list, y_axis, x_axis, ax=ax, cbarlabel=measure[idx], cmap="RdYlGn")
        annotate_heatmap(im, data=measure_list, valfmt="{x:.3f}")
        ax.invert_yaxis()
        ax.title.set_text(measure[idx])
        ax.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=False, top=False,
                       labeltop=False)
        fig.tight_layout()
        plt.savefig("results/" + measure[idx] + "_heatmap.png")

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    colors = ['r' if x == 0 else 'b' for x in y_train]
    axs[0].scatter(x_train[:, 0], x_train[:, 1], s=40, c=colors, alpha=0.8, cmap="RdYlBu", edgecolor="white")
    axs[0].title.set_text("Training Data")
    axs[0].set_aspect('equal', adjustable='box')
    colors = ['r' if x == 0 else 'b' for x in y_test]
    axs[1].scatter(x_test[:, 0], x_test[:, 1], s=40, c=colors, alpha=0.8, cmap="RdYlBu", edgecolor="white")
    axs[1].title.set_text("Test Data")
    axs[1].set_aspect('equal', adjustable='box')
    plt.savefig("results/test_train.png")


run_effectiveness_exp(dataset="data/adult/adult.data",
                      columns_to_scale=['X1', 'X3', 'X5', 'X11', 'X12', 'X13'],
                      columns_to_encode=['X2', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X14'],
                      label="Y")

# run_effectiveness_exp(dataset="data/data_banknote_authentication.txt",
#                       columns_to_scale=['X1', 'X2', 'X3', 'X4'],
#                       columns_to_encode=[],
#                       label="Y")

# run_effectiveness_exp(dataset="data/credit.csv",
#                       columns_to_scale=['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11'],
#                       columns_to_encode=['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20',
#                                          'X21', 'X22', 'X23'],
#                       label="Y", columns_to_drop=['ID'])

# run_synthetic_effectiveness_exp()
