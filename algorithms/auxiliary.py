__author__ = 'kunal'

import numpy as np
import pandas as pd

#import pylab as P
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split

def initialise():
    data_frame = pd.read_csv('../data/train.csv', header=0)
    return data_frame

def initialise_train(dates):
    if not dates:
        data_frame = pd.read_csv('../data/train.csv', header=0)
    elif dates:
        data_frame = pd.read_csv('../data/train.csv', header=0, parse_dates=['Dates'])
        data_frame['Year'] = data_frame['Dates'].map(lambda x: x.year)
        data_frame['Week'] = data_frame['Dates'].map(lambda x: x.week)
        data_frame['Hour'] = data_frame['Dates'].map(lambda x: x.hour)

    # Change string categories to integer classifiers
    # 1. determine all values
    Categories = list(enumerate(sorted(np.unique(data_frame['Category']))))
    Descriptions = list(enumerate(sorted(np.unique(data_frame['Descript']))))
    DaysOfWeeks = list(enumerate(sorted(np.unique(data_frame['DayOfWeek']))))
    PdDistricts = list(enumerate(sorted(np.unique(data_frame['PdDistrict']))))
    Resolutions = list(enumerate(sorted(np.unique(data_frame['Resolution']))))
    # 2. set up dictionaries
    CategoriesDict = {name: i for i, name in Categories}
    DescriptionsDict = {name: i for i, name in Descriptions}
    DaysOfWeeksDict = {name: i for i, name in DaysOfWeeks}
    PdDistrictsDict = {name: i for i, name in PdDistricts}
    ResolutionsDict = {name: i for i, name in Resolutions}
    # 3. Convert all strings to int
    data_frame.Category = data_frame.Category.map(lambda x: CategoriesDict[x]).astype(int)
    data_frame.Descript = data_frame.Descript.map(lambda x: DescriptionsDict[x]).astype(int)
    data_frame.DayOfWeek = data_frame.DayOfWeek.map(lambda x: DaysOfWeeksDict[x]).astype(int)
    data_frame.PdDistrict = data_frame.PdDistrict.map(lambda x: PdDistrictsDict[x]).astype(int)
    data_frame.Resolution = data_frame.Resolution.map(lambda x: ResolutionsDict[x]).astype(int)
    # rounding off location coordinates to 2 decimal points
    data_frame.X = data_frame.X.map(lambda x: "%.2f" % round(x, 2)).astype(float)
    data_frame.Y = data_frame.Y.map(lambda x: "%.2f" % round(x, 2)).astype(float)

    return data_frame

def initialise_test(dates):
    if not dates:
        data_frame = pd.read_csv('../data/test.csv', header=0)
    elif dates:
        data_frame = pd.read_csv('../data/test.csv', header=0, parse_dates=['Dates'])
        data_frame['Year'] = data_frame['Dates'].map(lambda x: x.year)
        data_frame['Week'] = data_frame['Dates'].map(lambda x: x.week)
        data_frame['Hour'] = data_frame['Dates'].map(lambda x: x.hour)

    # Change string categories to integer classifiers
    PdDistricts = list(enumerate(sorted(np.unique(data_frame['PdDistrict']))))
    DaysOfWeeks = list(enumerate(sorted(np.unique(data_frame['DayOfWeek']))))
    PdDistrictsDict = {name: i for i, name in PdDistricts}
    DaysOfWeeksDict = {name: i for i, name in DaysOfWeeks}
    data_frame.PdDistrict = data_frame.PdDistrict.map(lambda x: PdDistrictsDict[x]).astype(int)
    data_frame.DayOfWeek = data_frame.DayOfWeek.map(lambda x: DaysOfWeeksDict[x]).astype(int)
    # rounding off location coordinates to 2 decimal points
    data_frame.X = data_frame.X.map(lambda x: "%.2f" % round(x, 2)).astype(float)
    data_frame.Y = data_frame.Y.map(lambda x: "%.2f" % round(x, 2)).astype(float)
    return data_frame


# TODO: Fill missing values if any
# Compute mean of a column and fill missing values
def compute_mean(data_frame, column):
    columnName = str(column)
    meanValue = data_frame[columnName].dropna().mean()
    if len(data_frame.column[data_frame.column.isnull()]) > 0:
        data_frame.loc[(data_frame.column.isnull()), columnName] = meanValue

def calc_avg(A):
    return sum(A) / float(len(A))

def plotLearningCurves(train, classifier):
    #P.show()
    X = train.values[:, 1::]
    y = train.values[:, 0]

    train_sizes, train_scores, test_scores = learning_curve(
            classifier, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.xlabel("Training samples")
    plt.ylabel("Error Rate")
    plt.ylim((0, 1))
    plt.gca().invert_yaxis()
    plt.grid()

    # Plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")

    # Plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")

    # Draw the plot and reset the y-axis
    plt.draw()
    plt.gca().invert_yaxis()

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    classifier.fit(X_train, y_train)
    plt.show()


def k_fold_generator(X, y, k_fold):
    subset_size = len(X) / k_fold
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]

        yield X_train, y_train, X_valid, y_valid
