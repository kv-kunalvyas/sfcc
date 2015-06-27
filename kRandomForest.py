__author__ = 'kunal'
import numpy as np
import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier as rfc

trainDf = pd.read_csv('train.csv', header=0)#, parse_dates=['Dates'])

#trainDf['Year'] = trainDf['Dates'].map(lambda x: x.year)
#trainDf['Week'] = trainDf['Dates'].map(lambda x: x.week)
#trainDf['Hour'] = trainDf['Dates'].map(lambda x: x.hour)
# Dates
# Category -
# Descript -
# DayOfWeek .
# PdDistrict
# Resolution
# Address
# X
# Y

# TODO: Change string categories to integer classifiers
# determine all values
Categories = list(enumerate(sorted(np.unique(trainDf['Category']))))
Descriptions = list(enumerate(sorted(np.unique(trainDf['Descript']))))
DaysOfWeeks = list(enumerate(sorted(np.unique(trainDf['DayOfWeek']))))
PdDistricts = list(sorted(enumerate(np.unique(trainDf['PdDistrict']))))
Resolutions = list(sorted(enumerate(np.unique(trainDf['Resolution']))))
# set up dictionaries
CategoriesDict = {name: i for i, name in Categories}
DescriptionsDict = {name: i for i, name in Descriptions}
DaysOfWeeksDict = {name: i for i, name in DaysOfWeeks}
PdDistrictsDict = {name: i for i, name in PdDistricts}
ResolutionsDict = {name: i for i, name in Resolutions}
# Convert all strings to int
trainDf.Category = trainDf.Category.map(lambda x: CategoriesDict[x]).astype(int)
trainDf.Descript = trainDf.Descript.map(lambda x: DescriptionsDict[x]).astype(int)
trainDf.DayOfWeek = trainDf.DayOfWeek.map(lambda x: DaysOfWeeksDict[x]).astype(int)
trainDf.PdDistrict = trainDf.PdDistrict.map(lambda x: PdDistrictsDict[x]).astype(int)
trainDf.Resolution = trainDf.Resolution.map(lambda x: ResolutionsDict[x]).astype(int)

# TODO: Fill missing values if any
#Compute mean of a column and fill missing values
# def computeMean(column):
#     columnName = str(column)
#     meanValue = trainDf[columnName].dropna().mean()
#     if len(trainDf.column[ trainDf.column.isnull()]) > 0:
#         trainDf.loc[(trainDf.column.isnull()), columnName] = meanValue
#
# computeMean(Category)

trainDf = trainDf.drop(['Dates', 'Descript', 'Resolution', 'Address', 'X', 'Y'], axis=1)

# Test data
testDf = pd.read_csv('test.csv', header=0)
ids = testDf['Id'].values
testDf = testDf.drop(['Id', 'Dates', 'Address', 'X', 'Y'], axis=1)

PdDistricts = list(enumerate(sorted(np.unique(testDf['PdDistrict']))))
DaysOfWeeks = list(enumerate(sorted(np.unique(testDf['DayOfWeek']))))
PdDistrictsDict = {name: i for i, name in PdDistricts}
DaysOfWeeksDict = {name: i for i, name in DaysOfWeeks}
testDf.PdDistrict = testDf.PdDistrict.map(lambda x: PdDistrictsDict[x]).astype(int)
testDf.DayOfWeek = testDf.DayOfWeek.map(lambda x: DaysOfWeeksDict[x]).astype(int)

# TODO: Random Forest Algorithm
print list(trainDf.columns.values)
print list(testDf.columns.values)
# back to numpy format
trainData = trainDf.values
testData = testDf.values


print 'Training...'
forest = rfc(n_estimators=1)
forest = forest.fit(trainData[0::, 1::], trainData[0::])

print 'Predicting...'
output = forest.predict(testData).astype(int)
s = ""
for i in range(0, output.size, 1):
    for j in range(0, output[i][0], 1):

    #print output[i][0]

predictions_file = open("submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id",'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                           'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION',
                           'FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT',
                           'LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
                           'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY',
                           'SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY',
                           'SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
                           'WEAPON LAWS'])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
