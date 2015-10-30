__author__ = 'kunal'

import numpy as np
import pandas as pd
import decimal

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
