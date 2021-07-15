# ----------------------------//-------------------------IMPORT LIBRARIES

# Basic import
import os
import pickle
from numpy import ceil
import numpy as np
import pandas as pd
import itertools
import time
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# Plotting
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from IPython.display import SVG, display
from sklearn.tree import export_graphviz
import plotly.offline as py
import plotly.subplots as make_subplots
import plotly.figure_factory as ff  # visualization
import plotly.io as pio
import plotly.graph_objects as go
from yellowbrick.classifier import DiscriminationThreshold
from tabulate import tabulate


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, f_classif, SelectKBest

# Metrics
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import statsmodels.api as sm
from sklearn import tree

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from numpy import mean
from math import sqrt
from operator import itemgetter
import progressbar
import sys
from time import sleep

# Model Tuning
from bayes_opt import BayesianOptimization

# ----------------------------//-------------------------FUNCTIONS
# number of folds
n_splits = 5


# to view details of data
def detail_data(dataset, message):
    print(f'{message}: \n')
    print('Rows: ', dataset.shape[0])
    print('\n Number of features: ', dataset.shape[1])
    print('\n Features:')
    print(dataset.columns.tolist())
    print('\n Missing values:', dataset.isnull().sum().values.sum())
    print('\n Unique values:')
    print(dataset.nunique())
    print('\n Details of Dataset: \n')
    t = 0
    dt_features = dataset.columns
    for i in dt_features:
        t = t + 1
        print('\n{} - {}'.format(t, i))
        uni = dataset[i].unique()
        print('unique values: {}'.format(uni))
        print('Counts: {}'.format(len(uni)))
        print("-" * 100, '\n')


# to display Confusion matrix Chart
#
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.xticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # else:
    # print('Confusion matrix without normalization')

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_all_classes_distribution(df_value_counts, df_name, label):
    plt.figure(figsize=(8, 6))
    count = df_value_counts
    print(count)
    sns.set(style="darkgrid")
    sns.barplot(count.index, count.values, alpha=0.7)
    for index, data in enumerate(count):
        plt.text(index, data + 3, s=f"{data}", fontdict=dict(fontsize=12))
    plt.title('Frequency Distribution of {}'.format(df_name), fontsize=13)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(label, fontsize=12)
    plt.savefig('results_fig/' + df_name + '_Frequency_Distribution.png')
    # plt.show()


def save_model(model, out_dir):
    pickle.dump(model, open(out_dir, 'wb'))

def load_model(model_file):
    with open(model_file, 'rb') as file:
        return pickle.load(file)

def savefile_csv(target_file, saved_df):
    with open(target_file, 'w') as f_out:
        np.savetxt(f_out, saved_df, delimiter=',')


# remove special characters
# remove(filename, '\/:*?"<>|')
def remove(value, deletechars):
    for c in deletechars:
        value = value.replace(c, '')
    return value

#clean dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def convert(data):
    number = preprocessing.LabelEncoder()
    data['Month'] = number.fit_transform(data['Month'])
    data['VisitorType'] = number.fit_transform(data['VisitorType'])
    data = data.fillna(-9999)
    return data


def run_progressbar(ranges):
    for i in range(ranges):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        i_per = round(100 * float(i) / float(ranges), 0)
        # print('i: {} - i_per: {}'.format(i,i_per))
        text = '[{}] {}% row: {}'.format('=' * int(i_per), i_per, i)
        sys.stdout.write(text)
        # sys.stdout.write("[%-20s] %d%% %row: d%" % ('=' * int(i_per), i_per,i))
        sys.stdout.flush()
        sleep(0.25)
    print(" - Done")


def model_score(algorithm, testing_y, predictions, probabilities):
    # roc_auc_score
    print('Algorithm: ', type(algorithm).__name__)
    print('Classification report: \n', classification_report(testing_y, predictions))
    print('Accuracy score: ', accuracy_score(testing_y, predictions))

    model_roc_auc = roc_auc_score(testing_y, predictions)
    print('Area under Curve: \n', model_roc_auc, '\n')

    fpr, tpr, threshold = roc_curve(testing_y, probabilities[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.title('Confusion Maxtrix - {}'.format(type(algorithm).__name__))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # confusion matrix

    conf_matrix = confusion_matrix(testing_y, predictions)
    print("Confusion Matrix: \n", conf_matrix)
    # plt.figure()
    # plot_confusion_matrix(conf_matrix, classes, title='Confusion Maxtrix - {}'.format(type(algorithm).__name__))
    # plt.savefig('results_fig/ConfusionMatrix_{}'.format(type(algorithm).__name__))


def churn_prediction(algorithm, training_x, testing_x, training_y, testing_y, threshold_plot):
    # model 1
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    model_score(algorithm, testing_y, predictions, probabilities)

    if threshold_plot:
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x, training_y)
        visualizer.show()


def treeplot(classifier, cols, classnames):
    # plot decision tree

    graph = Source(tree.export_graphviz(classifier, out_file=None,
                                        rounded=True, proportion=False,
                                        feature_names=cols,
                                        precision=2,
                                        class_names=classnames,
                                        filled=True))
    # display(graph)


#########################################################
#        Model performance metrics                      #
#########################################################
# gives model report in dataframe
def model_report(model, training_x, testing_x, training_y, testing_y, name, kind, fold, datasetname):
    training_time_start = time.perf_counter()
    # print('----- Training starting time: {}'.format(training_time_start))
    model = model.fit(training_x, training_y)
    training_time_end = time.perf_counter()
    # print('----- Training ending time: {}'.format(training_time_end))
    training_time = training_time_end - training_time_start
    print('----------> Training time: {}\n'.format(round(training_time, 4)))

    if kind == 'SMOTE' or kind == 'RUS':
        save_model(model, 'model_2after_preprocessing_data/{}.pkl'.format(datasetname + '_' + name + '_' + str(fold)))
    else:
        save_model(model, 'model_1before_preprocessing_data/{}.pkl'.format(datasetname + '_' + name + '_' + str(fold)))

    prediction_time_start = time.perf_counter()
    predictions = model.predict(testing_x)
    prediction_time_end = time.perf_counter()
    prediction_time = prediction_time_end - prediction_time_start

    accuracy = accuracy_score(testing_y, predictions)
    recall = recall_score(testing_y, predictions)
    precision = precision_score(testing_y, predictions)
    roc_auc = roc_auc_score(testing_y, predictions)
    f1score = f1_score(testing_y, predictions)
    kappa_metric = cohen_kappa_score(testing_y, predictions)

    df = pd.DataFrame({"Fold:": [fold],
                       "Model": [name],
                       "Accuracy": [accuracy],
                       "Recall": [recall],
                       "Precision": [precision],
                       "f1-score": [f1score],
                       "Roc_auc": [roc_auc],
                       "Kappa_metric": [kappa_metric],
                       "Training time": [training_time],
                       "Prediction time": [prediction_time]
                       })
    return df


def euclidean_distance(a, b):
    return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(a, b)))


#########################################################
#        Compare model metrics                          #
#########################################################
def output_tracer(df, metric, color):
    tracer = go.Bar(y=df["Model"],
                    x=df[metric],
                    orientation="h", name=metric,
                    marker=dict(line=dict(width=.7), color=color)
                    )
    return tracer


def modelmetricsplot(df, title):
    layout = go.Layout(dict(title=title,
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            xaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       title="metric",
                                       zerolinewidth=1,
                                       ticklen=5, gridwidth=2),
                            yaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       zerolinewidth=1, ticklen=5, gridwidth=2),
                            margin=dict(l=250),
                            height=780
                            )
                       )
    trace1 = output_tracer(df, "Accuracy", "#6699FF")
    trace2 = output_tracer(df, 'Recall', "red")
    trace3 = output_tracer(df, 'Precision', "#33CC99")
    trace4 = output_tracer(df, 'f1-score', "lightgrey")
    trace5 = output_tracer(df, 'Roc_auc', "magenta")
    trace6 = output_tracer(df, 'Kappa_metric', "#FFCC99")

    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


########################################
# CONFUSION MATRIX                      #
########################################

def confmatplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber, kind, dataset_name, fold, labels):
    fig = plt.figure(figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber)))
    fig.set_facecolor("#F3F3F3")
    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        print('---', name)
        plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)

        if kind == 'Smote' or kind == 'Rus':
            model = modeldict[name][0].fit(df_train[1][modeldict[name][1]], target_train[1])
            predictions = model.predict(df_test[modeldict[name][1]])
            conf_matrix = confusion_matrix(target_test, predictions)
            sns.heatmap(conf_matrix, annot=True, fmt="d", square=True,
                        xticklabels=labels,
                        yticklabels=labels,
                        linewidths=2, linecolor="w", cmap="Set1")
            plt.title(name, color="b")
            plt.subplots_adjust(wspace=.3, hspace=.3)

        else:
            model = modeldict[name][0].fit(df_train[0][modeldict[name][1]], target_train[0])
            predictions = model.predict(df_test[modeldict[name][1]])
            conf_matrix = confusion_matrix(target_test, predictions)
            sns.heatmap(conf_matrix, annot=True, fmt="d", square=True,
                        xticklabels=labels,
                        yticklabels=labels,
                        linewidths=2, linecolor="w", cmap="Set1")
            plt.title(name, color="b")
            plt.subplots_adjust(wspace=.3, hspace=.3)
        plt.savefig('results_fig/' + dataset_name + '_confusionmatrix_' + kind + '_' + str(fold) + '.png')


########################################
# ROC - Curves for models               #
########################################
def rocplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber, kind, dataset_name, fold):
    fig = plt.figure(figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber)))
    fig.set_facecolor("#F3F3F3")
    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        print('---', name)
        qx = plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)
        if kind == 'Smote' or kind == 'Rus':
            model = modeldict[name][0].fit(df_train[1][modeldict[name][1]], target_train[1])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            fpr, tpr, thresholds = roc_curve(target_test, probabilities[:, 1])
            plt.plot(fpr, tpr, linestyle="dotted",
                     color="royalblue", linewidth=2,
                     label="AUC = " + str(np.around(roc_auc_score(target_test, predictions), 3)))
            plt.plot([0, 1], [0, 1], linestyle="dashed",
                     color="orangered", linewidth=1.5)
            plt.fill_between(fpr, tpr, alpha=.1)
            plt.fill_between([0, 1], [0, 1], color="b")
            plt.legend(loc="lower right",
                       prop={"size": 12})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xticks(np.arange(0, 1, .3))
            plt.yticks(np.arange(0, 1, .3))


        else:
            model = modeldict[name][0].fit(df_train[0][modeldict[name][1]], target_train[0])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            fpr, tpr, thresholds = roc_curve(target_test, probabilities[:, 1])
            plt.plot(fpr, tpr, linestyle="dotted",
                     color="royalblue", linewidth=2,
                     label="AUC = " + str(np.around(roc_auc_score(target_test, predictions), 3)))
            plt.plot([0, 1], [0, 1], linestyle="dashed",
                     color="orangered", linewidth=1.5)
            plt.fill_between(fpr, tpr, alpha=.1)
            plt.fill_between([0, 1], [0, 1], color="b")
            plt.legend(loc="lower right",
                       prop={"size": 12})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xticks(np.arange(0, 1, .3))
            plt.yticks(np.arange(0, 1, .3))
        plt.savefig('results_fig/' + dataset_name + '_roccurves_' + kind + '_' + str(fold) + '.png')


########################################
# Precision recall curves               #
########################################
def prcplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber, kind, dataset_name, fold):
    fig = plt.figure(figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber)))
    fig.set_facecolor("#F3F3F3")
    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        print('---', name)
        qx = plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)
        if kind == "Smote" or kind == 'Rus':
            model = modeldict[name][0].fit(df_train[1][modeldict[name][1]], target_train[1])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            recall, precision, thresholds = precision_recall_curve(target_test, probabilities[:, 1])
            plt.plot(recall, precision, linewidth=1.5,
                     label=("avg_pcn: " + str(np.around(average_precision_score(target_test, predictions), 3))))
            plt.plot([0, 1], [0, 0], linestyle="dashed")
            plt.fill_between(recall, precision, alpha=.1)
            plt.legend(loc="lower left", prop={"size": 10})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xlabel("recall", fontsize=7)
            plt.ylabel("precision", fontsize=7)
            plt.xlim([0.25, 1])
            plt.yticks(np.arange(0, 1, .3))

        else:
            model = modeldict[name][0].fit(df_train[0][modeldict[name][1]], target_train[0])
            probabilities = model.predict_proba(df_test[modeldict[name][1]])
            predictions = model.predict(df_test[modeldict[name][1]])

            recall, precision, thresholds = precision_recall_curve(target_test, probabilities[:, 1])
            plt.plot(recall, precision, linewidth=1.5,
                     label=("avg_pcn: " + str(np.around(average_precision_score(target_test, predictions), 3))))
            plt.plot([0, 1], [0, 0], linestyle="dashed")
            plt.fill_between(recall, precision, alpha=.1)
            plt.legend(loc="lower left", prop={"size": 10})
            qx.set_facecolor("w")
            plt.grid(True, alpha=.15)
            plt.title(name, color="b")
            plt.xlabel("recall", fontsize=7)
            plt.ylabel("precision", fontsize=7)
            plt.xlim([0.25, 1])
            plt.yticks(np.arange(0, 1, .3))
        plt.savefig('results_fig/' + dataset_name + '_precision_' + kind + '_' + str(fold) + '.png')


# -------------- Classifiers -----------------
# Baseline model
logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)

# LOGISTIC REGRESSION - SMOTE
logit_smote = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                 intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                 penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                 verbose=0, warm_start=False)

# LOGISTIC REGRESSION - Random Undersampling
logit_rus = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)

# base
decision_tree = DecisionTreeClassifier(max_depth=9,
                                       random_state=123,
                                       splitter="best",
                                       criterion="gini")

# smote
decision_tree_smote = DecisionTreeClassifier(max_depth=9,
                                             random_state=123,
                                             splitter="best",
                                             criterion="gini")
# rus
decision_tree_rus = DecisionTreeClassifier(max_depth=9,
                                           random_state=123,
                                           splitter="best",
                                           criterion="gini")

# base
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform')

# smote
knn_smote = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                                 weights='uniform')

# rus
knn_rus = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='uniform')

# base
rf = RandomForestClassifier(n_estimators=100, random_state=123,
                            max_depth=9, criterion="gini")

# smote
rf_smote = RandomForestClassifier(n_estimators=100, random_state=123,
                                  max_depth=9, criterion="gini")

# rus
rf_rus = RandomForestClassifier(n_estimators=100, random_state=123,
                                max_depth=9, criterion="gini")

# base
nb = GaussianNB(priors=None)

# smote
nb_smote = GaussianNB(priors=None)

# rus
nb_rus = GaussianNB(priors=None)

# LightGBM Classifier_base
lgbmc = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                       learning_rate=0.5, max_depth=7, min_child_samples=20,
                       min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                       n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                       reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                       subsample_for_bin=200000, subsample_freq=0)

# LightGBM Classifier_SMOTE
lgbmc_smote = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                             learning_rate=0.5, max_depth=7, min_child_samples=20,
                             min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                             n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                             reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                             subsample_for_bin=200000, subsample_freq=0)

# LightGBM Classifier_rus
lgbmc_rus = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                           learning_rate=0.5, max_depth=7, min_child_samples=20,
                           min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                           n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                           reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                           subsample_for_bin=200000, subsample_freq=0)

# XGBoost Classifier_base
xgc = XGBClassifier(base_score=0.5, booster='gbtree',
                    colsample_bylevel=1, colsample_bytree=1,
                    gamma=0, learning_rate=0.9,
                    max_delta_step=0, max_depth=10,
                    min_child_weight=1, n_estimators=100,
                    n_jobs=1, nthread=None,
                    objective='binary:logistic',
                    random_state=0, reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=1,
                    seed=None, subsample=1, eval_metric='logloss')

# XGBoost Classifier_smote
xgc_smote = XGBClassifier(base_score=0.5, booster='gbtree',
                          colsample_bylevel=1, colsample_bytree=1,
                          gamma=0, learning_rate=0.9,
                          max_delta_step=0, max_depth=10,
                          min_child_weight=1, n_estimators=100,
                          n_jobs=1, nthread=None,
                          objective='binary:logistic',
                          random_state=0, reg_alpha=0,
                          reg_lambda=1, scale_pos_weight=1,
                          seed=None, subsample=1, eval_metric='logloss')

# XGBoost Classifier_rus
xgc_rus = XGBClassifier(base_score=0.5, booster='gbtree',
                        colsample_bylevel=1, colsample_bytree=1,
                        gamma=0, learning_rate=0.9,
                        max_delta_step=0, max_depth=10,
                        min_child_weight=1, n_estimators=100,
                        n_jobs=1, nthread=None,
                        objective='binary:logistic',
                        random_state=0, reg_alpha=0,
                        reg_lambda=1, scale_pos_weight=1,
                        seed=None, subsample=1, eval_metric='logloss')

# Gaussian Process Classifier
gpc = GaussianProcessClassifier(random_state=124)

# Gaussian Process Classifier_smote
gpc_smote = GaussianProcessClassifier(random_state=124)

# Gaussian Process Classifier_rus
gpc_rus = GaussianProcessClassifier(random_state=124)

# AdaBoost Classifier_base
adac = AdaBoostClassifier(random_state=124)

# AdaBoost Classifier_smote
adac_smote = AdaBoostClassifier(random_state=124)

# AdaBoost Classifier_rus
adac_rus = AdaBoostClassifier(random_state=124)

# GradientBoosting Classifier_base
gbc = GradientBoostingClassifier(random_state=124)

# GradientBoosting Classifier_smote
gbc_smote = GradientBoostingClassifier(random_state=124)

# GradientBoosting Classifier_rus
gbc_rus = GradientBoostingClassifier(random_state=124)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()

# Linear Discriminant Analysis_smote
lda_smote = LinearDiscriminantAnalysis()

# Linear Discriminant Analysis_rus
lda_rus = LinearDiscriminantAnalysis()

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()

# Quadratic Discriminant Analysis_smote
qda_smote = QuadraticDiscriminantAnalysis()

# Quadratic Discriminant Analysis_rus
qda_rus = QuadraticDiscriminantAnalysis()

# Multi-layer Perceptron Classifier
mlp = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

# Multi-layer Perceptron Classifier_smote
mlp_smote = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

# Multi-layer Perceptron Classifier_rus
mlp_rus = MLPClassifier(alpha=1, max_iter=1000, random_state=124)

# Bagging Classifier
bgc = BaggingClassifier(random_state=124)

# Bagging Classifier_smote
bgc_smote = BaggingClassifier(random_state=124)

# Bagging Classifier_rus
bgc_rus = BaggingClassifier(random_state=124)

# Support vector classifier using linear hyper plane
svc_lin = SVC(C=1.0, kernel='linear', probability=True, random_state=124)

# Support vector classifier using linear hyper plane_smote
svc_lin_smote = SVC(C=1.0, kernel='linear', probability=True, random_state=124)

# Support vector classifier using linear hyper plane_rus
svc_lin_rus = SVC(C=1.0, kernel='linear', probability=True, random_state=124)

# support vector classifier using non-linear hyper plane ("rbf")
svc_rbf = SVC(C=10.0, kernel='rbf', gamma=0.1, probability=True, random_state=124)

# support vector classifier using non-linear hyper plane ("rbf")_smote
svc_rbf_smote = SVC(C=10.0, kernel='rbf', gamma=0.1, probability=True, random_state=124)

# support vector classifier using non-linear hyper plane ("rbf")_rus
svc_rbf_rus = SVC(C=10.0, kernel='rbf', gamma=0.1, probability=True, random_state=124)

# ----------------------------//-------------------------PREPROCESSING DATASET

# 1 - WA_Fn-UseC_-Telco-Customer-Churn #########################
#     - Du doan hanh vi de giu chan khach hang                 #
#     - Nhung khach hang bo di - goi la Churn                  #
#     - Khach hang su dung cac dich vu nhu: phone, internet,v.v#
################################################################

#---------------------------- Read file
filename = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
churn_data = pd.read_csv(filename, header=0, engine='python')
# remove special characters
remove(filename, '\/:*?"<>|')

#---------------------------- View data info
print('Ìnfomation of Telco Customer Churn dataset \n'.format(churn_data.info()))
print('\n Discription of Telco Customer Churn dataset: \n', churn_data.describe())

# xem ti le mat can bang - Churn / Non Churn
churn_data.loc[:, 'Churn_2'] = churn_data['Churn']
churn_data['Churn_2'].replace({"Yes": "Churn", "No": "Not Churn"}, inplace=True)  # temp col
churn_counts = churn_data.Churn_2.value_counts()

plot_all_classes_distribution(churn_counts, 'Telco Customer Churn', 'Churn / Not Churn Customers')

#---------------------------- Preprocessing
####### Change data type of 'TotalCharges' columns"
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
print(churn_data.isnull().sum())
churn_data = churn_data.reset_index()[churn_data.columns]
churn_features = list(churn_data.columns)
number_churn_features = len(churn_features)
print("Number of churn data feature: {} - Features: \n {}".format(number_churn_features, churn_features))
detail_data(churn_data,'Overview of Churn dataset')

# define columns
churn_id_col = ['customerID']
churn_target_col = ['Churn']

# new churn dataset with dropping CustomID column
churn_data = churn_data.drop("customerID", axis=1)
churn_data = churn_data.drop('Churn_2', axis=1)

# encode the label of Churn Column in churn_data2
churn_data['Churn'].replace({"Yes": 1, "No": 0}, inplace=True)

# delete the null Values in churn_data2
churn_data.dropna(inplace=True)

replace_cols = ['OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies']

for i in replace_cols:
    churn_data[i].replace({'No internet service': 'No'}, inplace=True)

churn_data['SeniorCitizen'].replace({0: "No", 1: "Yes"}, inplace=True)
churn_data['MultipleLines'].replace({'No phone service': 'No'}, inplace=True)
detail_data(churn_data,'Churn data after replacing some columns')

# Convert all features to dummy/indicator variables (bien gia/chi so)
churn_data.TotalCharges.astype("float")
churn_data = pd.get_dummies(churn_data)
churn_data_features = list(churn_data.columns)

# colunms of churn data
churn_cols = [i for i in churn_data.columns if i not in churn_id_col + churn_target_col]

# number of levels in feature to be a categorical feature
nlevels = 6

# categorical columns
churn_cat_cols = churn_data.nunique()[churn_data.nunique() < nlevels].keys().tolist()
churn_cat_cols = [x for x in churn_cat_cols if x not in churn_target_col]

# numerical cols
churn_num_cols = [x for x in churn_data.columns if x not in churn_cat_cols + churn_target_col + churn_id_col]

# binary columns with 2 values
churn_bin_cols = churn_data.nunique()[churn_data.nunique() == 2].keys().tolist()

# columns more than 2 values
churn_multi_cols = [i for i in churn_cat_cols if i not in churn_bin_cols]

# churn || non churn customer
churn = churn_data[churn_data['Churn'] == 1]
non_churn = churn_data[churn_data['Churn'] == 0]

# split data => Train - Test
churn_train, churn_test = train_test_split(churn_data, test_size=.4, random_state=111)

trainsize = churn_train.shape[0]
combine = pd.concat((churn_train,churn_test),sort=False)

#Duplicating columns for multi value columns
combine = pd.get_dummies(data = combine, columns=churn_multi_cols)

#Separating the train and test datasets
churn_train = combine[:trainsize]
churn_test = combine[trainsize:]
print('Trainsize: {}'.format(trainsize))
#detail_data(churn_train,'Overview of train Churn dataset')


# ------X-Y_data
churn_X = churn_data[churn_cols]
churn_Y = churn_data[churn_target_col]

# Scaling Numerical columns
std = StandardScaler()
churn_scaled = std.fit_transform(churn_data[churn_num_cols])
churn_scaled = pd.DataFrame(churn_scaled, columns=churn_num_cols)

churn_scaled_test = std.transform(churn_test[churn_num_cols])
churn_scaled_test = pd.DataFrame(churn_scaled_test, columns=churn_num_cols)

# dropping original values - merging scaled values for numerical columns

churn_data_og = churn_data.copy()
churn_data = churn_data.drop(columns=churn_num_cols, axis=1)
churn_data = churn_data.merge(churn_scaled, left_index=True, right_index=True, how="left")

churn_test_og = churn_test.copy()
churn_test = churn_test.drop(columns=churn_num_cols, axis=1)
churn_test = churn_test.merge(churn_scaled_test, left_index=True, right_index=True, how="left")


# variable summary

churn_summary = (churn_data_og[[i for i in churn_data_og.columns]].describe().transpose().reset_index())
churn_summary = churn_summary.rename(columns={"index": "feature"})
churn_summary = np.around(churn_summary, 3)

val_lst = [churn_summary['feature'],
           churn_summary['count'],
           churn_summary['mean'],
           churn_summary['std'],
           churn_summary['min'],
           churn_summary['25%'],
           churn_summary['50%'],
           churn_summary['75%'],
           churn_summary['max']]

trace = go.Table(header=dict(values=churn_summary.columns.tolist(),
                             line=dict(color=['#506784']),
                             fill=dict(color=['#119DFF'])),
                 cells=dict(values=val_lst,
                            line=dict(color=['#506784']),
                            fill=dict(color=['lightgrey', '#F5F8FF'])),
                 columnwidth=[200, 60, 100, 100, 60, 60, 80, 80, 80])

layout = go.Layout(dict(title='Training variable summary'))
figure = go.Figure(data=[trace], layout=layout)
py.plot(figure, filename='results_fig/Training_variable_summary.html')


# Correlation matrix
churn_correlation = churn_data.corr()

# tick labels
churn_matrix_cols = churn_correlation.columns.tolist()

# convert to array
churn_corr_array = np.array(churn_correlation)

# plot
trace = go.Heatmap(z=churn_corr_array,
                   x=churn_matrix_cols,
                   y=churn_matrix_cols,
                   colorscale="Viridis",
                   colorbar=dict(title='Pearson Correlation coefficients', titleside='right'))
layout = go.Layout(dict(title="Correlation Matrix",
                        autosize=False,
                        height=720,
                        width=800,
                        margin=dict(r=0, l=210, t=25, b=210),
                        yaxis=dict(tickfont=dict(size=9)),
                        xaxis=dict(tickfont=dict(size=9))))
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='results_fig/Pearson_correlation_coefficients.html')



# -------------------------------------


classes = ["Churn", "Not Churn"]


#######################################################
#        Model performances over the training dataset  #
########################################################

# defining the studied or used independent features (columns) as well the target
# churn_cols
# churn_target_cols

#splitting data

#shuffle the data before creating the subsamples
#churn_train = churn_train.sample(frac=1)



x_train, x_test, y_train, y_test = train_test_split(churn_train[churn_cols],
                                                    churn_train[churn_target_col],
                                                    test_size=.25,
                                                    random_state=111)

# oversampling minority class using smote
# --------------- SMOTE -------------
# Randomly pick a point from the minority class.
# Compute the k-nearest neighbors (for some pre-specified k) for this point.
# Add k new points somewhere between the chosen point and each of its neighbors

#training dataset:

#=== churn_train
# seperate majority and minority class <=> not_churn=0 and churn=1
churn_1 = churn_train[churn_train['Churn']==1] #minority-S
not_churn = churn_train[churn_train['Churn']==0] #majority-B

# ------X-Y_data
churn_train_X = churn_train.drop(['Churn'], axis=1)
churn_train_Y = churn_train['Churn']

x_test = churn_test.drop(['Churn'], axis=1)
y_test = churn_test['Churn']

#split to 5 folds
churn_kfold = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)

for fold, (train_index, test_index) in enumerate(churn_kfold.split(churn_train_X, churn_train_Y),1):
    print('\n----------- FOLD: {}\n'.format(fold))

    print('Train:',train_index, 'Test:', test_index)
    x_train_churn, x_test_churn = churn_train_X.iloc[train_index], churn_train_X.iloc[test_index]
    y_train_churn, y_test_churn  = churn_train_Y.iloc[train_index], churn_train_Y.iloc[test_index]

    smote = SMOTE(random_state=0)
    x_smote, y_smote = smote.fit_resample(x_train_churn, y_train_churn)
    x_smote = pd.DataFrame(data=x_smote, columns=churn_cols)
    y_smote = pd.DataFrame(data=y_smote, columns=churn_target_col)

    smote = SMOTE(sampling_strategy='minority')
    x_smote, y_smote = smote.fit_resample(x_train_churn, y_train_churn)
    x_smote = pd.DataFrame(data=x_smote, columns=churn_cols)
    y_smote = pd.DataFrame(data=y_smote, columns=churn_target_col)
    # -----------------------------------------------------------------------

    # identify noise and delete in training data after smote
    # Outlier Detection and Removal
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_smote)

    # select all rows that are not outliers
    mask = yhat != -1
    x_smote_train, y_smote_train = x_smote.iloc[mask, :], y_smote.iloc[mask]
    # summarize the shape of the updated training dataset
    print('\nx_smote: {} - y_smote: {}\n'.format(x_smote_train.shape, y_smote_train.shape))
    x_smote_2, y_smote_2 = smote.fit_resample(x_smote_train, y_smote_train)
    x_smote_r = pd.DataFrame(data=x_smote_2, columns=churn_cols)
    y_smote_r = pd.DataFrame(data=y_smote_2, columns=churn_target_col)

    # apply Random undersampling
    rus = RandomUnderSampler(random_state=0)
    rus.fit(x_smote_r, y_smote_r)
    x_rus, y_rus = rus.fit_resample(x_smote_r, y_smote_r)

    # putting all the model names, model classes and the used columns in a dictionary
    # Before preprocessing data
    models = {'Logistic_Baseline': [logit, churn_cols],
              'Decision Tree': [decision_tree, churn_cols],
              'KNN Classifier': [knn, churn_cols],
              'Random Forest': [rf, churn_cols],
              'Naive Bayes': [nb, churn_cols],
              'LGBMClassifier': [lgbmc, churn_cols],
              'XGBoost Classifier': [xgc, churn_cols],
              'Gaussian Process': [gpc, churn_cols],
              'AdaBoost': [adac, churn_cols],
              'GradientBoost': [gbc, churn_cols],
              'LDA': [lda, churn_cols],
              'QDA': [qda, churn_cols],
              'MLP Classifier': [mlp, churn_cols],
              'Bagging Classifier': [bgc, churn_cols],
              'SVM_linear': [svc_lin, churn_cols],
              'SVM_rbf': [svc_rbf, churn_cols]
              }

    # Applying SMOTE (smote)
    models_smote = {'Logistic_SMOTE': [logit_smote, churn_cols],
                    'Decision Tree_SMOTE': [decision_tree_smote, churn_cols],
                    'KNN Classifier_SMOTE': [knn_smote, churn_cols],
                    'Random Forest_SMOTE': [rf_smote, churn_cols],
                    'Naive Bayes_SMOTE': [nb_smote, churn_cols],
                    'LGBMClassifier_SMOTE': [lgbmc_smote, churn_cols],
                    'XGBoost Classifier_SMOTE': [xgc_smote, churn_cols],
                    'Gaussian Process_SMOTE': [gpc_smote, churn_cols],
                    'AdaBoost_SMOTE': [adac_smote, churn_cols],
                    'GradientBoost_SMOTE': [gbc_smote, churn_cols],
                    'LDA_SMOTE': [lda_smote, churn_cols],
                    'QDA_SMOTE': [qda_smote, churn_cols],
                    'MLP Classifier_SMOTE': [mlp_smote, churn_cols],
                    'Bagging Classifier_SMOTE': [bgc_smote, churn_cols],
                    'SVM_linear_SMOTE': [svc_lin_smote, churn_cols],
                    'SVM_rbf_SMOTE': [svc_rbf_smote, churn_cols]
                    }

    # Applying Random Undersampling after Over-sampling SMOTE (rus)
    models_rus = {'Logistic_RUS': [logit_rus, churn_cols],
                  'Decision Tree_RUS': [decision_tree_rus, churn_cols],
                  'KNN Classifier_RUS': [knn_rus, churn_cols],
                  'Random Forest_RUS': [rf_rus, churn_cols],
                  'Naive Bayes_RUS': [nb_rus, churn_cols],
                  'LGBM Classifier_RUS': [lgbmc_rus, churn_cols],
                  'XGBoost Classifier_RUS': [xgc_rus, churn_cols],
                  'Gaussian Process_RUS': [gpc_rus, churn_cols],
                  'AdaBoost_RUS': [adac_rus, churn_cols],
                  'GradientBoost_RUS': [gbc_rus, churn_cols],
                  'LDA_RUS': [lda_rus, churn_cols],
                  'QDA_RUS': [qda_rus, churn_cols],
                  'MLP Classifier_RUS': [mlp_rus, churn_cols],
                  'Bagging Classifier_RUS': [bgc_rus, churn_cols],
                  'SVM_linear_RUS': [svc_lin_rus, churn_cols],
                  'SVM_rbf_RUS': [svc_rbf_rus, churn_cols]
                  }

    # outputs for all models over the training dataset
    model_performances_train = pd.DataFrame()
    print('Outputs for all models over the training dataset')

    kinds = ['SMOTE', 'RUS', 'Base']
    for kind in kinds:
        print(kind)
        if kind == 'SMOTE':
            for name in models_smote:
                print("--- ",name)
                model_performances_train = model_performances_train.append(model_report(models_smote[name][0],
                                                                                        x_smote[models_smote[name][1]],
                                                                                        x_test_churn[models_smote[name][1]],
                                                                                        y_smote, y_test_churn, name, 'SMOTE',fold,'Churn'),
                                                                           ignore_index=True)
        elif kind == 'RUS':
            for name in models_rus:
                print("--- ",name)
                model_performances_train = model_performances_train.append(model_report(models_rus[name][0],
                                                                                        x_rus[models_rus[name][1]],
                                                                                        x_test_churn[models_rus[name][1]],
                                                                                        y_rus, y_test_churn, name, 'RUS',fold,'Churn'),
                                                                           ignore_index=True)
        else:
            for name in models:
                print("--- ",name)
                model_performances_train = model_performances_train.append(model_report(models[name][0],
                                                                                        x_train_churn[models[name][1]],
                                                                                        x_test_churn[models[name][1]],
                                                                                        y_train_churn, y_test_churn, name, 'Base',fold,'Churn'),
                                                                           ignore_index=True)

    ########################################
    # MODEL PERFORMANCES TRAIN                 #
    ########################################
    table_train = ff.create_table(np.round(model_performances_train, 4))
    table_train.show()
    filename_ = 'results_fig/Churn_Metrics_Table_'+str(fold)+'.html'
    py.iplot(table_train,filename=filename_)

    ##### modelmetricsplot(df=model_performances_train, title="Model performances over the training dataset")

    kinds = ["Baseline", "Smote", "Rus"]
    for kind in kinds:

        labels = ['Not Churn', 'Churn']

        if kind == 'Rus':
            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_churn_rus"))
            print('########################################')
            confmatplot(modeldict=models_rus, df_train=[x_train_churn, x_rus], df_test=x_test_churn,
                        target_train=[y_train_churn, y_rus], target_test=y_test_churn, figcolnumber=3, kind=kind,
                        dataset_name='Churn',fold=fold, labels= labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_churn_rus"))
            print('########################################')
            rocplot(modeldict=models_rus, df_train=[x_train_churn, x_rus], df_test=x_test_churn,
                    target_train=[y_train_churn, y_rus], target_test=y_test_churn, figcolnumber=3, kind=kind, dataset_name='Churn',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_churn_rus"))
            print('########################################')
            prcplot(modeldict=models_rus, df_train=[x_train_churn, x_rus], df_test=x_test_churn,
                    target_train=[y_train_churn, y_rus], target_test=y_test_churn, figcolnumber=3, kind=kind, dataset_name='Churn',fold=fold)
        elif kind == 'Smote':
            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_churn_smote"))
            print('########################################')
            confmatplot(modeldict=models_smote, df_train=[x_train_churn, x_smote], df_test=x_test_churn,
                        target_train=[y_train_churn, y_smote], target_test=y_test_churn, figcolnumber=3, kind=kind,
                        dataset_name='Churn',fold=fold, labels= labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_churn_smote"))
            print('########################################')
            rocplot(modeldict=models_smote, df_train=[x_train_churn, x_smote], df_test=x_test_churn,
                    target_train=[y_train_churn, y_smote], target_test=y_test_churn, figcolnumber=3, kind=kind,
                    dataset_name='Churn',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_churn_smote"))
            print('########################################')
            prcplot(modeldict=models_smote, df_train=[x_train_churn, x_smote], df_test=x_test_churn,
                    target_train=[y_train_churn, y_smote], target_test=y_test_churn, figcolnumber=3, kind=kind,
                    dataset_name='Churn',fold=fold)
        else:

            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_churn"))
            print('########################################')
            confmatplot(modeldict=models, df_train=[x_train_churn, x_smote], df_test=x_test_churn,
                        target_train=[y_train_churn, y_smote], target_test=y_test_churn, figcolnumber=3, kind=kind,
                        dataset_name='Churn',fold=fold, labels=labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_churn"))
            print('########################################')
            rocplot(modeldict=models, df_train=[x_train_churn, x_smote], df_test=x_test_churn,
                    target_train=[y_train_churn, y_smote], target_test=y_test_churn, figcolnumber=3, kind=kind,
                    dataset_name='Churn',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {}# - model: {}#'.format(kind, "models_churn"))
            print('########################################')
            prcplot(modeldict=models, df_train=[x_train_churn, x_smote], df_test=x_test_churn,
                    target_train=[y_train_churn, y_smote], target_test=y_test_churn, figcolnumber=3, kind=kind,
                    dataset_name='Churn',fold=fold)

    print('\n ------------ Fold {} has finished.\n'.format(fold))



########################################################################## DATASETS ####
# 2 -online_shoppers_intension: Dự đoán hành động mua hàng của khách online ############
########################################################################################

# ---------------------------- Read file
file_shopper = 'data/online_shoppers_intention.csv'
shopper_data = pd.read_csv(file_shopper, header=0, engine='python')
# remove special characters
remove(file_shopper, '\/:*?"<>|')

print('\n---------------The shape of data')
print(shopper_data.shape)
detail_data(shopper_data, '---------------Info of Online shopper intension')
print(shopper_data.describe())
print(shopper_data.head())

print('---------------checking the percentage of missing data contains in all the columns')

missing_percentage = shopper_data.isnull().sum()/shopper_data.shape[0]
print(missing_percentage)

shopper_data.info()


# checking the Distribution of customers on Revenue


plt.rcParams['figure.figsize'] = (18, 7)
plt.subplot(1, 2, 1)
sns.countplot(shopper_data['Weekend'], palette = 'pastel')
plt.title('Buy or Not', fontsize = 30)
plt.xlabel('Revenue or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)

# checking the Distribution of customers on Weekend
plt.subplot(1, 2, 2)
sns.countplot(shopper_data['Weekend'], palette = 'inferno')
plt.title('Purchase on Weekends', fontsize = 30)
plt.xlabel('Weekend or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.savefig('results_fig/shopper_DistributionOfCustomersOnRevenueAndWeekend.png')
plt.show()


# --------------------------------------------------------------------


print('\n-------- Count new visitors and return visitors')
print(shopper_data.VisitorType.value_counts())

print('\n---------plotting a pie chart for difference of visitors and browsers')

plt.rcParams['figure.figsize'] = (18, 7)
size = [10551, 1694, 85]
colors = ['violet', 'magenta', 'pink']
labels = "Returning Visitor", "New_Visitor", "Others"
explode = [0, 0, 0.1]
plt.subplot(1, 2, 1)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Different Visitors', fontsize = 30)
plt.axis('off')
plt.legend()

# plotting a pie chart for browsers

size = [7961, 2462, 736, 467,174, 163, 300]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'blue']
labels = "2", "1","4","5","6","10","others"

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle = 90)
plt.title('Different Browsers', fontsize = 30)
plt.axis('off')
plt.legend()
plt.savefig('results_fig/shopper_pie_chart_visitor_browser.png')
plt.show()


# -----------------------------------------------------------------------


print('\n-------------- visualizing the distribution of customers around the Region')
plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
plt.hist(shopper_data['TrafficType'], color = 'lightgreen')
plt.title('Distribution of diff Traffic',fontsize = 30)
plt.xlabel('TrafficType Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

# visualizing the distribution of customers around the Region

plt.subplot(1, 2, 2)
plt.hist(shopper_data['Region'], color = 'lightblue')
plt.title('Distribution of Customers',fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.savefig('results_fig/shopper_DistributionOfCustomersAroundTheRegion.png')
plt.show()


# -----------------------------------------------------------------------


print('\n-------------- the no. of OSes each user is having')
print(shopper_data.OperatingSystems.value_counts())
print('\n ------- the months with most no.of customers visiting the online shopping sites')
print(shopper_data.Month.value_counts())

print('\n ------ view the distribution of months with the no of customers visting')
# plotting a pie chart for different number of OSes users have.

size = [6601, 2585, 2555, 478, 111]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen']
labels = "2", "1","3","4","others"
explode = [0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.subplot(1, 2, 1)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('OSes Users have', fontsize = 14)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

# plotting a pie chart for share of special days

size = [3364, 2998, 1907, 1727, 549, 448, 433, 432, 288, 184]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'magenta', 'lightblue', 'lightgreen', 'violet']
labels = "May", "November", "March", "December", "October", "September", "August", "July", "June", "February"
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Special Days', fontsize = 14)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()
plt.show()
plt.savefig('results_fig/shoper_TimeOfEachUser_NumOfCustomerVisitPerMonth.png')


# -----------------------------------------------------------------------


print('\n--------- Duration of related products vs Revenue')
plt.rcParams['figure.figsize'] = (18, 15)

plt.subplot(2, 2, 1)
sns.boxenplot(shopper_data['Revenue'], shopper_data['Informational_Duration'], palette = 'rainbow')
plt.title('Info. duration vs Revenue', fontsize = 14)
plt.xlabel('Info. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)

plt.subplot(2, 2, 2)
sns.boxenplot(shopper_data['Revenue'], shopper_data['Administrative_Duration'], palette = 'pastel')
plt.title('Admn. duration vs Revenue', fontsize = 14)
plt.xlabel('Admn. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)

plt.subplot(2, 2, 3)
sns.boxenplot(shopper_data['Revenue'], shopper_data['ProductRelated_Duration'], palette = 'dark')
plt.title('Product Related duration vs Revenue', fontsize = 14)
plt.xlabel('Product Related duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)

print('\n--------The Exit Rate vs Revenue')
plt.subplot(2, 2, 4)
sns.boxenplot(shopper_data['Revenue'], shopper_data['ExitRates'], palette = 'spring')
plt.title('ExitRates vs Revenue', fontsize = 14)
plt.xlabel('ExitRates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)

plt.show()
plt.savefig('results_fig/shopper_SomeInfovsRevenua.png')


# -----------------------------------------------------------------------


print('\n--------- Page value vs Revenue')
plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)
sns.stripplot(shopper_data['Revenue'], shopper_data['PageValues'], palette = 'autumn')
plt.title('PageValues vs Revenue', fontsize = 14)
plt.xlabel('PageValues', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)

# bounce rates vs revenue
plt.subplot(1, 2, 2)
sns.stripplot(shopper_data['Revenue'], shopper_data['BounceRates'], palette = 'magma')
plt.title('Bounce Rates vs Revenue', fontsize = 14)
plt.xlabel('Boune Rates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.savefig('results_fig/shopper_PageValuevsRevenue.png')
plt.show()


# -----------------------------------------------------------------------


print('\n --------- weekend vs Revenue')
df = pd.crosstab(shopper_data['Weekend'], shopper_data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
plt.title('Weekend vs Revenue', fontsize = 14)
plt.savefig('results_fig/shopper_WeekendvsRevenue.png')
plt.show()

print('\n----- Traffic Type vs Revenue')
df = pd.crosstab(shopper_data['TrafficType'], shopper_data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightpink', 'yellow'])
plt.title('Traffic Type vs Revenue', fontsize = 14)
plt.savefig('results_fig/shopper_TrafficvsRevenue.png')
plt.show()

print('\n-------visitor type vs revenue')

df = pd.crosstab(shopper_data['VisitorType'], shopper_data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightgreen', 'green'])
plt.title('Visitor Type vs Revenue', fontsize = 14)
plt.savefig('results_fig/shopper_VisitorvsRevenue.png')
plt.show()

print('\n--------Region vs Revenue')
df = pd.crosstab(shopper_data['Region'], shopper_data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightblue', 'blue'])
plt.title('Region vs Revenue', fontsize = 14)
plt.savefig('results_fig/shopper_RegionvsRevenue.png')
plt.show()


#Linear Model Plot
plt.rcParams['figure.figsize'] = (20, 10)

sns.lmplot(x = 'Administrative', y = 'Informational', data = shopper_data, x_jitter = 0.05)
plt.title('LM Plot between Admistrative and Information', fontsize = 15)
plt.savefig('results_fig/shopper_LinearModel.png')
plt.show()


# -----------------------------------------------------------------------


### Multi-Variate Analysis

#Month vs PageValue - Revenue // how month and pagevalue change as Revenue changes
plt.rcParams['figure.figsize'] = (18, 15)
plt.subplot(2, 2, 1)
sns.boxplot(x = shopper_data['Month'], y = shopper_data['PageValues'], hue = shopper_data['Revenue'], palette = 'inferno')
plt.title('Mon. vs PageValues w.r.t. Rev.', fontsize = 14)


# month vs exit rates wrt revenue
plt.subplot(2, 2, 2)
sns.boxplot(x = shopper_data['Month'], y = shopper_data['ExitRates'], hue = shopper_data['Revenue'], palette = 'Reds')
plt.title('Mon. vs ExitRates w.r.t. Rev.', fontsize = 14)

# month vs bouncerates wrt revenue
plt.subplot(2, 2, 3)
sns.boxplot(x = shopper_data['Month'], y = shopper_data['BounceRates'], hue = shopper_data['Revenue'], palette = 'Oranges')
plt.title('Mon. vs BounceRates w.r.t. Rev.', fontsize = 14)

# visitor type vs exit rates w.r.t revenue
plt.subplot(2, 2, 4)
sns.boxplot(x = shopper_data['VisitorType'], y = shopper_data['BounceRates'], hue = shopper_data['Revenue'], palette = 'Purples')
plt.title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 14)
plt.savefig('results_fig/shopper_MultiVariateAnalysis.png')
plt.show()



# -----------------------------------------------------------------------


# visitor type vs exit rates w.r.t revenue
plt.rcParams['figure.figsize'] = (18, 15)
plt.subplot(2, 2, 1)
sns.violinplot(x = shopper_data['VisitorType'], y = shopper_data['ExitRates'], hue = shopper_data['Revenue'], palette = 'rainbow')
plt.title('Visitors vs ExitRates wrt Rev.', fontsize = 14)

# visitor type vs exit rates w.r.t revenue
plt.subplot(2, 2, 2)
sns.violinplot(x = shopper_data['VisitorType'], y = shopper_data['PageValues'], hue = shopper_data['Revenue'], palette = 'gnuplot')
plt.title('Visitors vs PageValues wrt Rev.', fontsize = 14)

# region vs pagevalues w.r.t. revenue
plt.subplot(2, 2, 3)
sns.violinplot(x = shopper_data['Region'], y = shopper_data['PageValues'], hue = shopper_data['Revenue'], palette = 'Greens')
plt.title('Region vs PageValues wrt Rev.', fontsize = 14)

#region vs exit rates w.r.t. revenue
plt.subplot(2, 2, 4)
sns.violinplot(x = shopper_data['Region'], y = shopper_data['ExitRates'], hue = shopper_data['Revenue'], palette = 'spring')
plt.title('Region vs Exit Rates w.r.t. Revenue', fontsize = 14)
plt.savefig('results_fig/shopper_VisitortypeExitrateRevenue.png')
plt.show()


# -----------------------------------------------------------------------

# Inputing Missing Values with 0



shopper_data = convert(shopper_data)
#shopper_data.fillna(999, inplace = True)

# checking the no. of null values in data after imputing the missing values
shopper_data.isnull().sum().sum()

#-----------------------------------------------------------------------

######### Data Preprocessing #######

####### One Hot and Label Encoding

# one hot encoding
shopper_data_1 = pd.get_dummies(shopper_data)
print(shopper_data_1.columns)


# label encoding of revenue

shopper_labelEncoder = LabelEncoder()
shopper_data['Revenue'] = shopper_labelEncoder.fit_transform(shopper_data['Revenue'])
shopper_data_counts = shopper_data['Revenue'].value_counts()
plot_all_classes_distribution(shopper_data_counts,"Online Shopper Intension","Not Buy / Buy")
shopper_target_col = ['Revenue']
# colunms of data
shopper_cols = [i for i in shopper_data.columns if i not in shopper_target_col]

# getting dependent and independent variables
temp_data = shopper_data_1

shopper_train, shopper_test = train_test_split(temp_data, test_size=.25, random_state=12)

shopper_train_size = shopper_train.shape[0]
print(shopper_train_size)

shopper_comb = pd.concat((shopper_train,shopper_test),sort=False)


shopper_train = shopper_comb[:shopper_train_size]
shopper_test = shopper_comb[shopper_train_size:]


# removing the target column revenue from x
x_train_shopper_data = shopper_train.drop(['Revenue'],axis=1)
y_train_shopper_data = shopper_train['Revenue']

x_test_shopper_data = shopper_test.drop(['Revenue'],axis=1)
y_test_shopper_data = shopper_test['Revenue']

# checking the shapes
#print("Shape of x:", x_train_shopper_data.shape)
#print("Shape of y:", y_train_shopper_data.shape)

# ------------------------------------SPLITTING THE DATA

x_train_shopper_data, x_test_shopper_data, y_train_shopper_data, y_test_shopper_data = train_test_split(x_shopper_data,
                                                                                                        y_shopper_data,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)


#checking the shapes of train & test dataset
print("Shape of x_train :", x_train_shopper_data.shape)
print("Shape of y_train :", y_train_shopper_data.shape)
print("Shape of x_test :", x_test_shopper_data.shape)
print("Shape of y_test :", y_test_shopper_data.shape)


#split - 5 folds
k_fold = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)

for fold, (train_index, test_index) in enumerate(k_fold.split(x_train_shopper_data, y_train_shopper_data),1):
    print('----------- FOLD: {}'.format(fold))
    print('Train: ', train_index, 'Test: ', test_index)
    x_train_shopper_fold, x_test_shopper_fold = x_train_shopper_data.iloc[train_index], x_train_shopper_data.iloc[test_index]
    y_train_shopper_fold, y_test_shopper_fold = y_train_shopper_data.iloc[train_index], y_train_shopper_data.iloc[test_index]

    ###### smote train set
    smote_shopper_data = SMOTE(sampling_strategy='minority')
    x_smote_shopper_data, y_smote_shopper_data = smote_shopper_data.fit_resample(x_train_shopper_fold,
                                                                                 y_train_shopper_fold)
    x_smote_shopper_data = pd.DataFrame(data=x_smote_shopper_data, columns=shopper_cols)
    y_smote_shopper_data = pd.DataFrame(data=y_smote_shopper_data, columns=shopper_target_col)

    # identify noise and delete in training data after smote
    # Outlier Detection and Removal
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_smote_shopper_data)

    # select all rows that are not outliers
    mask = yhat != -1
    x_smote_shopper_data_train, y_smote_shopper_data_train = x_smote_shopper_data.iloc[mask, :], \
                                                             y_smote_shopper_data.iloc[mask]
    # summarize the shape of the updated training dataset
    print(x_smote_shopper_data_train.shape, y_smote_shopper_data_train.shape)
    x_smote_shopper_data_2, y_smote_shopper_data_2 = smote_shopper_data.fit_resample(x_smote_shopper_data_train,
                                                                                     y_smote_shopper_data_train)
    x_smote_shopper_data_r = pd.DataFrame(data=x_smote_shopper_data_2, columns=shopper_cols)
    y_smote_shopper_data_r = pd.DataFrame(data=y_smote_shopper_data_2, columns=shopper_target_col)

    # apply Random undersampling
    rus = RandomUnderSampler(random_state=0)
    rus.fit(x_smote_shopper_data_r, y_smote_shopper_data_r)
    x_rus_shopper_data, y_rus_shopper_data = rus.fit_resample(x_smote_shopper_data_r, y_smote_shopper_data_r)

    # ---------------------------------------------------------------------------

    # putting all the model names, model classes and the used columns in a dictionary
    # Before preprocessing data
    models_shopper = {'Logistic_Baseline': [logit, shopper_cols],
                      'Decision Tree': [decision_tree, shopper_cols],
                      'KNN Classifier': [knn, shopper_cols],
                      'Random Forest': [rf, shopper_cols],
                      'Naive Bayes': [nb, shopper_cols],
                      'LGBMClassifier': [lgbmc, shopper_cols],
                      'XGBoost Classifier': [xgc, shopper_cols],
                      'Gaussian Process': [gpc, shopper_cols],
                      'AdaBoost': [adac, shopper_cols],
                      'GradientBoost': [gbc, shopper_cols],
                      'LDA': [lda, shopper_cols],
                      'QDA': [qda, shopper_cols],
                      'MLP Classifier': [mlp, shopper_cols],
                      'Bagging Classifier': [bgc, shopper_cols],
                      'SVM_linear': [svc_lin, shopper_cols],
                      'SVM_rbf': [svc_rbf, shopper_cols]
                      }

    # Applying SMOTE (smote)
    models_shopper_smote = {'Logistic_SMOTE': [logit_smote, shopper_cols],
                            'Decision Tree_SMOTE': [decision_tree_smote, shopper_cols],
                            'KNN Classifier_SMOTE': [knn_smote, shopper_cols],
                            'Random Forest_SMOTE': [rf_smote, shopper_cols],
                            'Naive Bayes_SMOTE': [nb_smote, shopper_cols],
                            'LGBMClassifier_SMOTE': [lgbmc_smote, shopper_cols],
                            'XGBoost Classifier_SMOTE': [xgc_smote, shopper_cols],
                            'Gaussian Process_SMOTE': [gpc_smote, shopper_cols],
                            'AdaBoost_SMOTE': [adac_smote, shopper_cols],
                            'GradientBoost_SMOTE': [gbc_smote, shopper_cols],
                            'LDA_SMOTE': [lda_smote, shopper_cols],
                            'QDA_SMOTE': [qda_smote, shopper_cols],
                            'MLP Classifier_SMOTE': [mlp_smote, shopper_cols],
                            'Bagging Classifier_SMOTE': [bgc_smote, shopper_cols],
                            'SVM_linear_SMOTE': [svc_lin_smote, shopper_cols],
                            'SVM_rbf_SMOTE': [svc_rbf_smote, shopper_cols]
                            }

    # Applying Random Undersampling after Over-sampling SMOTE (rus)
    models_shopper_rus = {'Logistic_RUS': [logit_rus, shopper_cols],
                          'Decision Tree_RUS': [decision_tree_rus, shopper_cols],
                          'KNN Classifier_RUS': [knn_rus, shopper_cols],
                          'Random Forest_RUS': [rf_rus, shopper_cols],
                          'Naive Bayes_RUS': [nb_rus, shopper_cols],
                          'LGBM Classifier_RUS': [lgbmc_rus, shopper_cols],
                          'XGBoost Classifier_RUS': [xgc_rus, shopper_cols],
                          'Gaussian Process_RUS': [gpc_rus, shopper_cols],
                          'AdaBoost_RUS': [adac_rus, shopper_cols],
                          'GradientBoost_RUS': [gbc_rus, shopper_cols],
                          'LDA_RUS': [lda_rus, shopper_cols],
                          'QDA_RUS': [qda_rus, shopper_cols],
                          'MLP Classifier_RUS': [mlp_rus, shopper_cols],
                          'Bagging Classifier_RUS': [bgc_rus, shopper_cols],
                          'SVM_linear_RUS': [svc_lin_rus, shopper_cols],
                          'SVM_rbf_RUS': [svc_rbf_rus, shopper_cols]
                          }

    # outputs for all models over the training dataset
    model_performances_train_shopper_data = pd.DataFrame()
    print('Outputs for all models over the training dataset')
    kinds = ['SMOTE', 'RUS', 'Base']
    for kind in kinds:
        print(kind)
        if kind == 'SMOTE':
            for name in models_shopper_smote:
                print("--- ",name)
                model_performances_train_shopper_data = model_performances_train_shopper_data.append(
                    model_report(models_shopper_smote[name][0],
                                 x_smote_shopper_data[models_shopper_smote[name][1]],
                                 x_test_shopper_fold[models_shopper_smote[name][1]],
                                 y_smote_shopper_data, y_test_shopper_fold, name, 'SMOTE',fold,'Shopper'), ignore_index=True)
        elif kind == 'RUS':
            for name in models_shopper_rus:
                print("--- ",name)
                model_performances_train_shopper_data = model_performances_train_shopper_data.append(
                    model_report(models_shopper_rus[name][0],
                                 x_rus_shopper_data[models_shopper_rus[name][1]],
                                 x_test_shopper_fold[models_shopper_rus[name][1]],
                                 y_rus_shopper_data, y_test_shopper_fold, name, 'RUS',fold,'Shopper'), ignore_index=True)
        else:
            for name in models_shopper:
                print("--- ",name)
                model_performances_train_shopper_data = model_performances_train_shopper_data.append(
                    model_report(models_shopper[name][0],
                                 x_train_shopper_fold[models_shopper[name][1]],
                                 x_test_shopper_fold[models_shopper[name][1]],
                                 y_train_shopper_fold, y_test_shopper_fold, name, 'Base',fold,'Shopper'), ignore_index=True)

    ########################################
    # MODEL PERFORMANCES TRAIN                 #
    ########################################
    table_train = ff.create_table(np.round(model_performances_train_shopper_data, 4))
    table_train.show()
    filename_ = 'results_fig/Shopper_Metrics_Table_'+str(fold)+'.html'
    py.iplot(table_train,filename=filename_)

    #modelmetricsplot(df=model_performances_train_shopper_data, title="Model performances over the training dataset")

    kinds = ["Baseline", "Smote", "Rus"]
    for kind in kinds:

        labels = ["Not Buy", "Buy"]

        if kind == 'Rus':
            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_shopper_rus"))
            print('########################################')
            confmatplot(modeldict=models_shopper_rus, df_train=[x_train_shopper_fold, x_rus_shopper_data],
                        df_test=x_test_shopper_fold,
                        target_train=[y_train_shopper_fold, y_rus_shopper_data], target_test=y_test_shopper_fold,
                        figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold, labels= labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_shopper_rus"))
            print('########################################')
            rocplot(modeldict=models_shopper_rus, df_train=[x_train_shopper_fold, x_rus_shopper_data],
                    df_test=x_test_shopper_fold,
                    target_train=[y_train_shopper_fold, y_rus_shopper_data], target_test=y_test_shopper_fold,
                    figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_shopper_rus"))
            print('########################################')
            prcplot(modeldict=models_shopper_rus, df_train=[x_train_shopper_fold, x_rus_shopper_data],
                    df_test=x_test_shopper_fold,
                    target_train=[y_train_shopper_fold, y_rus_shopper_data], target_test=y_test_shopper_fold,
                    figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold)
        elif kind == 'Smote':
            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_shopper_smote"))
            print('########################################')
            confmatplot(modeldict=models_shopper_smote, df_train=[x_train_shopper_fold, x_smote_shopper_data],
                        df_test=x_test_shopper_fold,
                        target_train=[y_train_shopper_fold, y_smote_shopper_data], target_test=y_test_shopper_fold,
                        figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold, labels=labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_shopper_smote"))
            print('########################################')
            rocplot(modeldict=models_shopper_smote, df_train=[x_train_shopper_fold, x_smote_shopper_data],
                    df_test=x_test_shopper_fold,
                    target_train=[y_train_shopper_fold, y_smote_shopper_data], target_test=y_test_shopper_fold,
                    figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_shopper_smote"))
            print('########################################')
            prcplot(modeldict=models_shopper_smote, df_train=[x_train_shopper_fold, x_smote_shopper_data],
                    df_test=x_test_shopper_fold,
                    target_train=[y_train_shopper_fold, y_smote_shopper_data], target_test=y_test_shopper_fold,
                    figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold)
        else:

            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_shopper"))
            print('########################################')
            confmatplot(modeldict=models_shopper, df_train=[x_train_shopper_fold, x_smote_shopper_data],
                        df_test=x_test_shopper_fold,
                        target_train=[y_train_shopper_fold, y_smote_shopper_data], target_test=y_test_shopper_fold,
                        figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold, labels=labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_shopper"))
            print('########################################')
            rocplot(modeldict=models_shopper, df_train=[x_train_shopper_fold, x_smote_shopper_data],
                    df_test=x_test_shopper_fold,
                    target_train=[y_train_shopper_fold, y_smote_shopper_data], target_test=y_test_shopper_fold,
                    figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_shopper"))
            print('########################################')
            prcplot(modeldict=models_shopper, df_train=[x_train_shopper_fold, x_smote_shopper_data],
                    df_test=x_test_shopper_fold,
                    target_train=[y_train_shopper_fold, y_smote_shopper_data], target_test=y_test_shopper_fold,
                    figcolnumber=3, kind=kind, dataset_name='shopper',fold=fold)

    print('Fold {}:\n'.format(fold))



#### DATASETS ####
# 3 -Credit Card: giao dich the tin dung ############
#     - Trung binh cac khoan giao dich la 88 USD    #
#     - Khong co gia tri NULL                       #
#     - Hau het cac giao dich la Non-Fraud (99.83%) #
#      con lai la Fraud (0.17%)                     #
#####################################################

print("\n ========== credit card ========== \n")
file_creditcard = 'data/creditcard.csv'
creditcard_data = pd.read_csv(file_creditcard, header=0, engine='python')
print(creditcard_data.head())
print(creditcard_data.columns)
print(creditcard_data.describe())

# Good No Null Values!
creditcard_data.isnull().sum().max()

# remove special characters
remove(file_creditcard, '\/:*?"<>|')

#set index for columns
creditcard_data = pd.DataFrame(creditcard_data)


print('\n---------------The shape of data')
detail_data(creditcard_data, '---------------Info of Credit Fraud detector')
print(creditcard_data.describe())
print(creditcard_data.head())
print('Dataset - Original')
print(creditcard_data.info())

print('---------------checking the percentage of missing data contains in all the columns')

missing_percentage = creditcard_data.isnull().sum()/creditcard_data.shape[0]
print(missing_percentage)

creditcard_target = ['Class']

# colunms of data
creditcard_cols = [i for i in creditcard_data.columns if i not in creditcard_target]

# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(creditcard_data['Class'].value_counts()[0]/len(creditcard_data) * 100,2), '% of the dataset')
print('Frauds', round(creditcard_data['Class'].value_counts()[1]/len(creditcard_data) * 100,2), '% of the dataset')

plot_all_classes_distribution(creditcard_data.Class.value_counts(),'CreditFraudDetector','No Frauds / Frauds')

amount_val = creditcard_data['Amount'].values
time_val = creditcard_data['Time'].values


fig, ax = plt.subplots(1, 2, figsize=(18,4))

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
#plt.savefig('results_fig/CreditFraudDetector_DistributionOfTransactionAmountandTime.png')
plt.show()


#SCALER

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

creditcard_data['scaled_amount'] = rob_scaler.fit_transform(creditcard_data['Amount'].values.reshape(-1,1))
creditcard_data['scaled_time'] = rob_scaler.fit_transform(creditcard_data['Time'].values.reshape(-1,1))

creditcard_data.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = creditcard_data['scaled_amount']
scaled_time = creditcard_data['scaled_time']

creditcard_data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
creditcard_data.insert(0, 'scaled_amount', scaled_amount)
creditcard_data.insert(1, 'scaled_time', scaled_time)

creditcard_data = creditcard_data.astype(np.float64)
fraud = creditcard_data.loc[creditcard_data['Class']==1]
notfraud = creditcard_data.loc[creditcard_data['Class']==0]

sns.relplot(x='scaled_amount', y = 'scaled_time', hue = 'Class', data = creditcard_data)
#plt.savefig('results_fig/CreditFraudDetector_VisualDistribution_scaled_AmountandTime.png')
plt.show()

# Amount and Time are Scaled!
print(creditcard_data.head())

# colunms of data
creditcard_cols_scaled = [i for i in creditcard_data.columns if i not in creditcard_target]

#shuffle the data before creating the subsamples
#creditcard_data = creditcard_data.sample(frac=1)

creditcard_train, creditcard_test = train_test_split(creditcard_data,test_size=.25, random_state=111, shuffle=True)
creditcard_train_size = creditcard_train.shape[0]
print("Train_size: ", creditcard_train_size)

credit_comb = pd.concat((creditcard_train,creditcard_test),sort=False)


creditcard_train = credit_comb[:creditcard_train_size]
creditcard_test = credit_comb[creditcard_train_size:]

# amount of fraud classes 492 rows.
#fraud_ = creditcard_data.loc[creditcard_data['Class'] == 1]
#non_fraud_ = creditcard_data.loc[creditcard_data['Class'] == 0][:492]

#creditcard_data = pd.concat([fraud_,non_fraud_])

#Shuffle dataframe rows
#creditcard_data = creditcard_data.sample(frac=1,random_state=42)
#print('subsample: \n',creditcard_data.head())


#delete null values
#creditcard_data.dropna(inplace = True)
#creditcard_data.reset_index()[creditcard_data.columns]
#print(creditcard_data.head())

#Splitting data
y_creditcard_data = creditcard_train["Class"]
x_creditcard_data = creditcard_train.drop("Class", axis=1)


#------------------------------------------------------------------------------
# putting all the model names, model classes and the used columns in a dictionary
#Before preprocessing data

models_credit = {'Logistic_Baseline': [logit, creditcard_cols_scaled],
                  'Decision Tree': [decision_tree, creditcard_cols_scaled],
                  'KNN Classifier': [knn, creditcard_cols_scaled],
                  'Random Forest': [rf, creditcard_cols_scaled],
                  'Naive Bayes': [nb, creditcard_cols_scaled],
                  'LGBMClassifier': [lgbmc, creditcard_cols_scaled],
                  'XGBoost Classifier': [xgc, creditcard_cols_scaled],
                  'Gaussian Process': [gpc, creditcard_cols_scaled],
                  'AdaBoost': [adac, creditcard_cols_scaled],
                  'GradientBoost': [gbc, creditcard_cols_scaled],
                  'LDA': [lda, creditcard_cols_scaled],
                  'QDA': [qda, creditcard_cols_scaled],
                  'MLP Classifier': [mlp, creditcard_cols_scaled],
                  'Bagging Classifier': [bgc, creditcard_cols_scaled],
                  'SVM_linear': [svc_lin, creditcard_cols_scaled],
                  'SVM_rbf': [svc_rbf, creditcard_cols_scaled]
                  }

#Applying SMOTE (smote)
models_credit_smote = {'Logistic_SMOTE': [logit_smote, creditcard_cols_scaled],
                        'Decision Tree_SMOTE': [decision_tree_smote, creditcard_cols_scaled],
                        'KNN Classifier_SMOTE': [knn_smote, creditcard_cols_scaled],
                        'Random Forest_SMOTE': [rf_smote, creditcard_cols_scaled],
                        'Naive Bayes_SMOTE': [nb_smote, creditcard_cols_scaled],
                        'LGBMClassifier_SMOTE': [lgbmc_smote, creditcard_cols_scaled],
                        'XGBoost Classifier_SMOTE': [xgc_smote, creditcard_cols_scaled],
                        'Gaussian Process_SMOTE': [gpc_smote, creditcard_cols_scaled],
                        'AdaBoost_SMOTE': [adac_smote, creditcard_cols_scaled],
                        'GradientBoost_SMOTE': [gbc_smote, creditcard_cols_scaled],
                        'LDA_SMOTE': [lda_smote, creditcard_cols_scaled],
                        'QDA_SMOTE': [qda_smote, creditcard_cols_scaled],
                        'MLP Classifier_SMOTE': [mlp_smote, creditcard_cols_scaled],
                        'Bagging Classifier_SMOTE': [bgc_smote, creditcard_cols_scaled],
                        'SVM_linear_SMOTE': [svc_lin_smote, creditcard_cols_scaled],
                        'SVM_rbf_SMOTE': [svc_rbf_smote, creditcard_cols_scaled]
                        }

#Applying Random Undersampling after Over-sampling SMOTE (rus)
models_credit_rus = {'Logistic_RUS': [logit_rus, creditcard_cols_scaled],
                      'Decision Tree_RUS': [decision_tree_rus, creditcard_cols_scaled],
                      'KNN Classifier_RUS': [knn_rus, creditcard_cols_scaled],
                      'Random Forest_RUS': [rf_rus, creditcard_cols_scaled],
                      'Naive Bayes_RUS': [nb_rus, creditcard_cols_scaled],
                      'LGBM Classifier_RUS': [lgbmc_rus, creditcard_cols_scaled],
                      'XGBoost Classifier_RUS': [xgc_rus, creditcard_cols_scaled],
                      'Gaussian Process_RUS': [gpc_rus, creditcard_cols_scaled],
                      'AdaBoost_RUS': [adac_rus, creditcard_cols_scaled],
                      'GradientBoost_RUS': [gbc_rus, creditcard_cols_scaled],
                      'LDA_RUS': [lda_rus, creditcard_cols_scaled],
                      'QDA_RUS': [qda_rus, creditcard_cols_scaled],
                      'MLP Classifier_RUS': [mlp_rus, creditcard_cols_scaled],
                      'Bagging Classifier_RUS': [bgc_rus, creditcard_cols_scaled],
                      'SVM_linear_RUS': [svc_lin_rus, creditcard_cols_scaled],
                      'SVM_rbf_RUS': [svc_rbf_rus, creditcard_cols_scaled]
                      }

#----------------------------------------------------------------------------------------------

x_train_creditcard_data, x_test_creditcard_data, y_train_creditcard_data, y_test_creditcard_data = train_test_split(x_creditcard_data,
                                                                                                                    y_creditcard_data,
                                                                                                                    test_size=0.25,
                                                                                                                    random_state=0)
print(x_train_creditcard_data.shape)
print(y_train_creditcard_data.shape)
#k = number of folds

k_fold = StratifiedKFold(n_splits=n_splits,random_state=None, shuffle=False)



for fold, (train_index, test_index) in enumerate(k_fold.split(x_train_creditcard_data, y_train_creditcard_data),1):
    print('----------- FOLD: {}'.format(fold))
    print('Train:', train_index, 'Test:', test_index)
    x_train_creditcard_fold, x_test_creditcard_fold = x_train_creditcard_data.iloc[train_index], x_train_creditcard_data.iloc[test_index]
    y_train_creditcard_fold, y_test_creditcard_fold = y_train_creditcard_data.iloc[train_index], y_train_creditcard_data.iloc[test_index]

    # Check the Distribution of the labels
    # Turn into an array
    #x_train_creditcard = x_train_creditcard.values
    #x_test_creditcard = x_test_creditcard.values
    #y_train_creditcard = y_train_creditcard.values
    #y_test_creditcard = y_test_creditcard.values

    #print('Fold {}:\n'.format(fold))

    print('----- SMOTE')
###### smote train set
    smote_creditcard_data = SMOTE(sampling_strategy='minority')
# perform SMOTE within each fold / avoiding train_test_split in favour of KFold
    x_smote_creditcard_data, y_smote_creditcard_data = smote_creditcard_data.fit_resample(x_train_creditcard_fold,
                                                                                          y_train_creditcard_fold)
    x_smote_creditcard_data = np.nan_to_num(x_smote_creditcard_data)
    x_smote_creditcard_data = pd.DataFrame(data=x_smote_creditcard_data, columns=creditcard_cols_scaled)
    y_smote_creditcard_data = np.nan_to_num(y_smote_creditcard_data)
    y_smote_creditcard_data = pd.DataFrame(data=y_smote_creditcard_data, columns=creditcard_target)

    print('-----Outlier detection and removal')
    # identify noise and delete in training data after smote
    # Outlier Detection and Removal
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_smote_creditcard_data)

    # select all rows that are not outliers
    mask = yhat != -1
    x_smote_creditcard_data_train, y_smote_creditcard_data_train = x_smote_creditcard_data.iloc[mask, :], \
                                                                   y_smote_creditcard_data.iloc[mask]
    # summarize the shape of the updated training dataset
    print(x_smote_creditcard_data_train.shape, y_smote_creditcard_data_train.shape)
    print('Updating training dataset')
    x_smote_creditcard_data_2, y_smote_creditcard_data_2 = smote_creditcard_data.fit_resample(x_smote_creditcard_data_train, y_smote_creditcard_data_train)
    x_smote_creditcard_data_r = pd.DataFrame(data=x_smote_creditcard_data_2, columns=creditcard_cols_scaled)
    y_smote_creditcard_data_r = pd.DataFrame(data=y_smote_creditcard_data_2, columns=creditcard_target)

    print('----- Random undersampling')
    # apply Random undersampling
    rus_credit = RandomUnderSampler(random_state=0)
    rus_credit.fit(x_smote_creditcard_data_r, y_smote_creditcard_data_r)
    x_rus_creditcard_data, y_rus_creditcard_data = rus_credit.fit_resample(x_smote_creditcard_data_r,
                                                                           y_smote_creditcard_data_r)

    # outputs for all models over the training dataset
    model_performances_train_creditcard_data = pd.DataFrame()

    print('Outputs for all models over the training dataset')
    kinds = ['SMOTE', 'RUS', 'Base']
    for kind in kinds:
        print('-- ',kind)
        if kind == 'SMOTE':
            for name in models_credit_smote:
                print('--- ',name)
                model_performances_train_creditcard_data = model_performances_train_creditcard_data.append(
                    model_report(models_credit_smote[name][0],
                                 x_smote_creditcard_data[models_credit_smote[name][1]],
                                 x_test_creditcard_fold[models_credit_smote[name][1]],
                                 y_smote_creditcard_data, y_test_creditcard_fold, name, 'SMOTE',fold,'Credit'), ignore_index=True)


        elif kind == 'RUS':
            for name in models_credit_rus:
                print('--- ',name)
                model_performances_train_creditcard_data = model_performances_train_creditcard_data.append(
                    model_report(models_credit_rus[name][0],
                                 x_rus_creditcard_data[models_credit_rus[name][1]],
                                 x_test_creditcard_fold[models_credit_rus[name][1]],
                                 y_rus_creditcard_data, y_test_creditcard_fold, name, 'RUS',fold,'Credit'), ignore_index=True)
        else:
            for name in models_credit:
                print('--- ',name)
                model_performances_train_creditcard_data = model_performances_train_creditcard_data.append(
                    model_report(models_credit[name][0],
                                 x_train_creditcard_fold[models_credit[name][1]],
                                 x_test_creditcard_fold[models_credit[name][1]],
                                 y_train_creditcard_fold, y_test_creditcard_fold, name, 'Base',fold,'Credit'), ignore_index=True)
    
    ########################################
    # MODEL PERFORMANCES TRAIN                 #
    ########################################
    table_train = ff.create_table(np.round(model_performances_train_creditcard_data, 4))
    table_train.show()
    filename_ = 'results_fig/Credit_Metrics_Table_'+str(fold)+'.ly'
    py.iplot(table_train,filename=filename_)

    

    kinds = ["Baseline", "Smote", "Rus"]
    for kind in kinds:

        labels = ['Not Fraud', 'Fraud']

        if kind == 'Rus':
            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit_rus"))
            print('########################################')
            confmatplot(modeldict=models_credit_rus, df_train=[x_train_creditcard_fold, x_rus_creditcard_data],
                        df_test=x_test_creditcard_fold,
                        target_train=[y_train_creditcard_fold, y_rus_creditcard_data], target_test=y_test_creditcard_fold,
                        figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold, labels = labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit_rus"))
            print('########################################')
            rocplot(modeldict=models_credit_rus, df_train=[x_train_creditcard_fold, x_rus_creditcard_data],
                    df_test=x_test_creditcard_fold,
                    target_train=[y_train_creditcard_fold, y_rus_creditcard_data], target_test=y_test_creditcard_fold,
                    figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_credit_rus"))
            print('########################################')
            prcplot(modeldict=models_credit_rus, df_train=[x_train_creditcard_fold, x_rus_creditcard_data],
                    df_test=x_test_creditcard_fold,
                    target_train=[y_train_creditcard_fold, y_rus_creditcard_data], target_test=y_test_creditcard_fold,
                    figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold)
        elif kind == 'Smote':



            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit_smote"))
            print('########################################')
            confmatplot(modeldict=models_credit_smote, df_train=[x_train_creditcard_fold, x_smote_creditcard_data],
                        df_test=x_test_creditcard_fold,
                        target_train=[y_train_creditcard_fold, y_smote_creditcard_data], target_test=y_test_creditcard_fold,
                        figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold, labels = labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit_smote"))
            print('########################################')
            rocplot(modeldict=models_credit_smote, df_train=[x_train_creditcard_fold, x_smote_creditcard_data],
                    df_test=x_test_creditcard_fold,
                    target_train=[y_train_creditcard_fold, y_smote_creditcard_data], target_test=y_test_creditcard_fold,
                    figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_credit_smote"))
            print('########################################')
            prcplot(modeldict=models_credit_smote, df_train=[x_train_creditcard_fold, x_smote_creditcard_data],
                    df_test=x_test_creditcard_fold,
                    target_train=[y_train_creditcard_fold, y_smote_creditcard_data], target_test=y_test_creditcard_fold,
                    figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold)
        else:


            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit"))
            print('########################################')
            confmatplot(modeldict=models_credit, df_train=[x_train_creditcard_fold, x_smote_creditcard_data],
                        df_test=x_test_creditcard_fold,
                        target_train=[y_train_creditcard_fold, y_smote_creditcard_data], target_test=y_test_creditcard_fold,
                        figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold, labels = labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit"))
            print('########################################')
            rocplot(modeldict=models_credit, df_train=[x_train_creditcard_fold, x_smote_creditcard_data],
                    df_test=x_test_creditcard_fold,
                    target_train=[y_train_creditcard_fold, y_smote_creditcard_data], target_test=y_test_creditcard_fold,
                    figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_credit"))
            print('########################################')
            prcplot(modeldict=models_credit, df_train=[x_train_creditcard_fold, x_smote_creditcard_data],
                    df_test=x_test_creditcard_fold,
                    target_train=[y_train_creditcard_fold, y_smote_creditcard_data], target_test=y_test_creditcard_fold,
                    figcolnumber=3, kind=kind, dataset_name='Credit',fold=fold)


#### DATASETS ####
# 4 -BankruptcyPrediction: Du doan pha san ###############################
#     - Dữ liệu thu thập từ Tạp chi kinh tế Đài Loan từ 1999 đến 2009    #
#     - Việc phá sản của công ty được xác định dựa trên các              #
#     quy định kinh doanh của Sở giạo dịch chứng khóan Đài Loan          #
#     - Target: Y - Bankrupt?: Class label: 0-Not Bankrupt, 1-Bankrupt   #
##########################################################################

print("\n ========== Bankruptcy Prediction ========== \n")
file_bankruptcy = 'data/BankruptcyPrediction.csv'
# remove special characters
remove(file_bankruptcy, '\/:*?"<>|')

bankruptcy_data = pd.read_csv(file_bankruptcy, header=0, engine='python')

#clean dataset
clean_dataset(bankruptcy_data)

# Good No Null Values!
bankruptcy_data.isnull().sum().max()


print(bankruptcy_data.head())
print(bankruptcy_data.columns)
print(bankruptcy_data.describe())

### Rename Column names
# X0 => target
column_names = ['Y']
column_names = column_names + ['X' + str(num) for num in range(1,len(bankruptcy_data.columns))]
column_names_df = pd.DataFrame({"Var_name": column_names,"Description": bankruptcy_data.columns})

bankruptcy_data.columns = column_names
bankruptcy_data.info(verbose = True,show_counts = True)

print(tabulate(column_names_df))

detail_data(bankruptcy_data, '---------------Info of Bankruptcy Prediction')

#preprocessing data
for int_column in bankruptcy_data.select_dtypes(include="int64"):
    print(bankruptcy_data[int_column].value_counts())
    print("\n")

# drop column C94 for being a useless feature, only value is 1
bankruptcy_data = bankruptcy_data.drop("X94",axis = "columns")



print('---------------checking the percentage of missing data contains in all the columns')

missing_percentage = bankruptcy_data.isnull().sum()/bankruptcy_data.shape[0]
print(missing_percentage)

bankruptcy_target = ['Y']

# colunms of data
bankruptcy_cols = column_names

print('Not Banktrupt', round(bankruptcy_data.Y.value_counts()[0]/len(bankruptcy_data)*100,2), '% of the dataset')
print('Bankrupt', round(bankruptcy_data.Y.value_counts()[1]/len(bankruptcy_data) * 100,2), '% of the dataset')

plot_all_classes_distribution(bankruptcy_data.Y.value_counts(),'BankruptcyPrediction', 'Not Bankrupt / Bankrupt')

#feature selection
from sklearn.feature_selection import VarianceThreshold
X = bankruptcy_data.drop(['Y'],axis=1)
y = bankruptcy_data['Y']
var_thres = VarianceThreshold(3.0)
var_thres.fit(X)

#get required features
required_features = [col for col in X.columns if col in X.columns[var_thres.get_support()]]

print("Result: {} required features - {}".format(len(required_features),required_features))
bankruptcy_data_new = bankruptcy_data[required_features]
bankruptcy_data_new.loc[:,'Y'] = y
bankruptcy_data_new.reset_index()
print(bankruptcy_data_new)

#SPLIT DATA

bankruptcy_train, bankruptcy_test = train_test_split(bankruptcy_data_new, test_size=.25, random_state=111, shuffle=True)
bankruptcy_train_size = bankruptcy_train.shape[0]
print("Train_size: ", bankruptcy_train_size)

bankruptcy_comb = pd.concat((bankruptcy_train,bankruptcy_test),sort=False)

bankruptcy_train = bankruptcy_comb[:bankruptcy_train_size]
bankruptcy_test = bankruptcy_comb[bankruptcy_train_size:]


#Splitting data
y_bankruptcy_data = bankruptcy_train["Y"]
x_bankruptcy_data = bankruptcy_train.drop("Y",axis=1)

bankruptcy_cols_new = required_features

#------------------------------------------------------------------------------
# putting all the model names, model classes and the used columns in a dictionary
#Before preprocessing data

models_bankruptcy = {'Logistic_Baseline': [logit, bankruptcy_cols_new],
                     'Decision Tree': [decision_tree, bankruptcy_cols_new],
                     'KNN Classifier': [knn, bankruptcy_cols_new],
                     'Random Forest': [rf, bankruptcy_cols_new],
                     'Naive Bayes': [nb, bankruptcy_cols_new],
                     'LGBMClassifier': [lgbmc, bankruptcy_cols_new],
                     'XGBoost Classifier': [xgc, bankruptcy_cols_new],
                     'Gaussian Process': [gpc, bankruptcy_cols_new],
                     'AdaBoost': [adac, bankruptcy_cols_new],
                     'GradientBoost': [gbc, bankruptcy_cols_new],
                     'LDA': [lda, bankruptcy_cols_new],
                     'QDA': [qda, bankruptcy_cols_new],
                     'MLP Classifier': [mlp, bankruptcy_cols_new],
                     'Bagging Classifier': [bgc, bankruptcy_cols_new],
                     'SVM_linear': [svc_lin, bankruptcy_cols_new],
                     'SVM_rbf': [svc_rbf, bankruptcy_cols_new]
                     }


#Applying SMOTE (smote)
models_bankruptcy_smote = {'Logistic_SMOTE': [logit_smote, bankruptcy_cols_new],
                           'Decision Tree_SMOTE': [decision_tree_smote, bankruptcy_cols_new],
                           'KNN Classifier_SMOTE': [knn_smote, bankruptcy_cols_new],
                           'Random Forest_SMOTE': [rf_smote, bankruptcy_cols_new],
                           'Naive Bayes_SMOTE': [nb_smote, bankruptcy_cols_new],
                           'LGBMClassifier_SMOTE': [lgbmc_smote, bankruptcy_cols_new],
                           'XGBoost Classifier_SMOTE': [xgc_smote, bankruptcy_cols_new],
                           'Gaussian Process_SMOTE': [gpc_smote, bankruptcy_cols_new],
                           'AdaBoost_SMOTE': [adac_smote, bankruptcy_cols_new],
                           'GradientBoost_SMOTE': [gbc_smote, bankruptcy_cols_new],
                           'LDA_SMOTE': [lda_smote, bankruptcy_cols_new],
                           'QDA_SMOTE': [qda_smote, bankruptcy_cols_new],
                           'MLP Classifier_SMOTE': [mlp_smote, bankruptcy_cols_new],
                           'Bagging Classifier_SMOTE': [bgc_smote, bankruptcy_cols_new],
                           'SVM_linear_SMOTE': [svc_lin_smote, bankruptcy_cols_new],
                           'SVM_rbf_SMOTE': [svc_rbf_smote, bankruptcy_cols_new]
                           }

#Applying Random Undersampling after Over-sampling SMOTE (rus)
models_bankruptcy_rus = {'Logistic_RUS': [logit_rus, bankruptcy_cols_new],
                         'Decision Tree_RUS': [decision_tree_rus, bankruptcy_cols_new],
                         'KNN Classifier_RUS': [knn_rus, bankruptcy_cols_new],
                         'Random Forest_RUS': [rf_rus, bankruptcy_cols_new],
                         'Naive Bayes_RUS': [nb_rus, bankruptcy_cols_new],
                         'LGBM Classifier_RUS': [lgbmc_rus, bankruptcy_cols_new],
                         'XGBoost Classifier_RUS': [xgc_rus, bankruptcy_cols_new],
                         'Gaussian Process_RUS': [gpc_rus, bankruptcy_cols_new],
                         'AdaBoost_RUS': [adac_rus, bankruptcy_cols_new],
                         'GradientBoost_RUS': [gbc_rus, bankruptcy_cols_new],
                         'LDA_RUS': [lda_rus, bankruptcy_cols_new],
                         'QDA_RUS': [qda_rus, bankruptcy_cols_new],
                         'MLP Classifier_RUS': [mlp_rus, bankruptcy_cols_new],
                         'Bagging Classifier_RUS': [bgc_rus, bankruptcy_cols_new],
                         'SVM_linear_RUS': [svc_lin_rus, bankruptcy_cols_new],
                         'SVM_rbf_RUS': [svc_rbf_rus, bankruptcy_cols_new]
                         }

#----------------------------------------------------------------------------------------------

x_train_bankruptcy_data, x_test_bankruptcy_data, y_train_bankruptcy_data, y_test_bankruptcy_data = train_test_split(x_bankruptcy_data,
                                                                                                                    y_bankruptcy_data,
                                                                                                                    test_size=.25,
                                                                                                                    random_state=42)
print(x_train_bankruptcy_data.shape)
print(y_train_bankruptcy_data.shape)
#k = number of folds
k_fold = StratifiedKFold(n_splits=n_splits,random_state=None, shuffle=False)


for fold, (train_index, test_index) in enumerate(k_fold.split(x_train_bankruptcy_data, y_train_bankruptcy_data), 1):
    print('----------- FOLD: {}'.format(fold))
    print('Train:', train_index, 'Test:', test_index)
    x_train_bankruptcy_fold, x_test_bankruptcy_fold = x_train_bankruptcy_data.iloc[train_index], x_train_bankruptcy_data.iloc[test_index]
    y_train_bankruptcy_fold, y_test_bankruptcy_fold = y_train_bankruptcy_data.iloc[train_index], y_train_bankruptcy_data.iloc[test_index]


    print('--------------------------- SMOTE')
    ###### smote train set
    smote_bankruptcy_data = SMOTE(sampling_strategy='minority')
    # perform SMOTE within each fold / avoiding train_test_split in favour of KFold
    x_smote_bankruptcy_data, y_smote_bankruptcy_data = smote_bankruptcy_data.fit_resample(x_train_bankruptcy_fold,
                                                                                          y_train_bankruptcy_fold)

    x_smote_bankruptcy_data = pd.DataFrame(data=x_smote_bankruptcy_data, columns=bankruptcy_cols_new)
    y_smote_bankruptcy_data = pd.DataFrame(data=y_smote_bankruptcy_data, columns=bankruptcy_target)





    print('-------------------------------------Outlier detection and removal')
    # identify noise and delete in training data after smote
    # Outlier Detection and Removal
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(x_smote_bankruptcy_data)

    # select all rows that are not outliers
    mask = yhat != -1
    x_smote_bankruptcy_data_train, y_smote_bankruptcy_data_train = x_smote_bankruptcy_data.iloc[mask, :], \
                                                                   y_smote_bankruptcy_data.iloc[mask]
    # summarize the shape of the updated training dataset
    print(x_smote_bankruptcy_data_train.shape, y_smote_bankruptcy_data_train.shape)
    print('----------------Updating training dataset')
    x_smote_bankruptcy_data_2, y_smote_bankruptcy_data_2 = smote_bankruptcy_data.fit_resample(x_smote_bankruptcy_data_train,
                                                                                              y_smote_bankruptcy_data_train)
    x_smote_bankruptcy_data_r = pd.DataFrame(data=x_smote_bankruptcy_data_2, columns=bankruptcy_cols_new)
    y_smote_bankruptcy_data_r = pd.DataFrame(data=y_smote_bankruptcy_data_2, columns=bankruptcy_target)

    print('-------------------------------- Random undersampling')
    # apply Random undersampling
    rus_bankruptcy = RandomUnderSampler(random_state=0)
    rus_bankruptcy.fit(x_smote_bankruptcy_data_r, y_smote_bankruptcy_data_r)
    x_rus_bankruptcy_data, y_rus_bankruptcy_data = rus_bankruptcy.fit_resample(x_smote_bankruptcy_data_r,
                                                                               y_smote_bankruptcy_data_r)

    # outputs for all models over the training dataset
    model_performances_train_bankruptcy_data = pd.DataFrame()
    print('----------------------------Outputs for all models over the training dataset')
    kinds = ['SMOTE', 'RUS', 'Base']
    for kind in kinds:

        
        print('-- ', kind)
        if kind == 'SMOTE':
            for name in models_bankruptcy_smote:
                print('--- ', name)
                model_performances_train_bankruptcy_data = model_performances_train_bankruptcy_data.append(
                    model_report(models_bankruptcy_smote[name][0],
                                 x_smote_bankruptcy_data[models_bankruptcy_smote[name][1]],
                                 x_test_bankruptcy_fold[models_bankruptcy_smote[name][1]],
                                 y_smote_bankruptcy_data, y_test_bankruptcy_fold, name, 'SMOTE', fold, 'Bankruptcy'),
                    ignore_index=True)


        elif kind == 'RUS':
            for name in models_bankruptcy_rus:
                print('--- ', name)
                model_performances_train_bankruptcy_data = model_performances_train_bankruptcy_data.append(
                    model_report(models_bankruptcy_rus[name][0],
                                 x_rus_bankruptcy_data[models_bankruptcy_rus[name][1]],
                                 x_test_bankruptcy_fold[models_bankruptcy_rus[name][1]],
                                 y_rus_bankruptcy_data, y_test_bankruptcy_fold, name, 'RUS', fold, 'Bankruptcy'),
                    ignore_index=True)
        else:
            for name in models_bankruptcy:
                print('--- ', name)
                model_performances_train_bankruptcy_data = model_performances_train_bankruptcy_data.append(
                    model_report(models_bankruptcy[name][0],
                                 x_train_bankruptcy_fold[models_bankruptcy[name][1]],
                                 x_test_bankruptcy_fold[models_bankruptcy[name][1]],
                                 y_train_bankruptcy_fold, y_test_bankruptcy_fold, name, 'Base', fold, 'Bankruptcy'),
                    ignore_index=True)

    ########################################
    # MODEL PERFORMANCES TRAIN             #
    ########################################
    table_train = ff.create_table(np.round(model_performances_train_bankruptcy_data, 4))
    table_train.show()
    filename_ = 'results_fig/Bankruptcy_Metrics_Table_' + str(fold) + '.ly'
    py.iplot(table_train, filename=filename_)

    kinds = ["Baseline", "Smote", "Rus"]
    for kind in kinds:
        labels = ['Not Bankrupt', 'Bankrupt']

        if kind == 'Rus':


            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_bankruptcy_rus"))
            print('########################################')
            confmatplot(modeldict=models_bankruptcy_rus, df_train=[x_train_bankruptcy_fold, x_rus_bankruptcy_data],
                        df_test=x_test_bankruptcy_fold,
                        target_train=[y_train_bankruptcy_fold, y_rus_bankruptcy_data],
                        target_test=y_test_bankruptcy_fold,
                        figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold, labels = labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_bankruptcy_rus"))
            print('########################################')
            rocplot(modeldict=models_bankruptcy_rus, df_train=[x_train_bankruptcy_fold, x_rus_bankruptcy_data],
                    df_test=x_test_bankruptcy_fold,
                    target_train=[y_train_bankruptcy_fold, y_rus_bankruptcy_data], target_test=y_test_bankruptcy_fold,
                    figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_credit_rus"))
            print('########################################')
            prcplot(modeldict=models_bankruptcy_rus, df_train=[x_train_bankruptcy_fold, x_rus_bankruptcy_data],
                    df_test=x_test_bankruptcy_fold,
                    target_train=[y_train_bankruptcy_fold, y_rus_bankruptcy_data], target_test=y_test_bankruptcy_fold,
                    figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold)

        elif kind == 'Smote':



            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_bankruptcy_smote"))
            print('########################################')
            confmatplot(modeldict=models_bankruptcy_smote, df_train=[x_train_bankruptcy_fold, x_smote_bankruptcy_data],
                        df_test=x_test_bankruptcy_fold,
                        target_train=[y_train_bankruptcy_fold, y_smote_bankruptcy_data],
                        target_test=y_test_bankruptcy_fold,
                        figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold, labels= labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_bankruptcy_smote"))
            print('########################################')
            rocplot(modeldict=models_bankruptcy_smote, df_train=[x_train_bankruptcy_fold, x_smote_bankruptcy_data],
                    df_test=x_test_bankruptcy_fold,
                    target_train=[y_train_bankruptcy_fold, y_smote_bankruptcy_data], target_test=y_test_bankruptcy_fold,
                    figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_bankruptcy_smote"))
            print('########################################')
            prcplot(modeldict=models_bankruptcy_smote, df_train=[x_train_bankruptcy_fold, x_smote_bankruptcy_data],
                    df_test=x_test_bankruptcy_fold,
                    target_train=[y_train_bankruptcy_fold, y_smote_bankruptcy_data], target_test=y_test_bankruptcy_fold,
                    figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold)

        else:


            print('########################################')
            print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_bankruptcy"))
            print('########################################')

            confmatplot(modeldict=models_bankruptcy, df_train=[x_train_bankruptcy_fold, x_smote_bankruptcy_data],
                        df_test=x_test_bankruptcy_fold,
                        target_train=[y_train_bankruptcy_fold, y_smote_bankruptcy_data],
                        target_test=y_test_bankruptcy_fold,
                        figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold, labels=labels)

            print('########################################')
            print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_bankruptcy"))
            print('########################################')
            rocplot(modeldict=models_bankruptcy, df_train=[x_train_bankruptcy_fold, x_smote_bankruptcy_data],
                    df_test=x_test_bankruptcy_fold,
                    target_train=[y_train_bankruptcy_fold, y_smote_bankruptcy_data], target_test=y_test_bankruptcy_fold,
                    figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold)

            print('########################################')
            print('# Precision recall curves------- {} - model: {}#'.format(kind, "models_bankruptcy"))
            print('########################################')
            prcplot(modeldict=models_bankruptcy, df_train=[x_train_bankruptcy_fold, x_smote_bankruptcy_data],
                    df_test=x_test_bankruptcy_fold,
                    target_train=[y_train_bankruptcy_fold, y_smote_bankruptcy_data], target_test=y_test_bankruptcy_fold,
                    figcolnumber=3, kind=kind, dataset_name='Bankruptcy', fold=fold)

