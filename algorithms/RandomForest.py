__author__ = 'kunal'

import csv
from sklearn.ensemble import RandomForestClassifier
import auxiliary
import warnings
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb

warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', Warning)

features = ['hour','dark','DayOfWeek','PdDistrict','StreetNo','X','Y']
features_non_numeric = ['DayOfWeek','PdDistrict']

# trainDf = auxiliary.initialise_train(True)
trainDf = pd.read_csv('../data/train.csv')
testDf = pd.read_csv('../data/test.csv')

trainDf['StreetNo'] = trainDf['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
trainDf['Address'] = trainDf['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
trainDf['hour'] = trainDf['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
trainDf['dark'] = trainDf['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)

trainDf = trainDf.drop(['Dates', 'Descript', 'Resolution', 'Address'], axis=1)

Categories = list(enumerate(sorted(np.unique(trainDf['Category']))))
CategoriesDict = {name: i for i, name in Categories}
trainDf.Category = trainDf.Category.map(lambda x: CategoriesDict[x]).astype(int)

#X, y = train_test_split(trainDf, train_size=.75)

# Test data #
#testDf = auxiliary.initialise_test(True)
testDf['StreetNo'] = testDf['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
testDf['Address'] = testDf['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
testDf['hour'] = testDf['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
testDf['dark'] = testDf['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)

ids = testDf['Id'].values
testDf = testDf.drop(['Id', 'Dates', 'Address'], axis=1)

le = LabelEncoder()
for col in features_non_numeric:
    le.fit(list(trainDf[col])+list(testDf[col]))
    trainDf[col] = le.transform(trainDf[col])
    testDf[col] = le.transform(testDf[col])

scaler = StandardScaler()
for col in features:
    scaler.fit(list(trainDf[col])+list(testDf[col]))
    trainDf[col] = scaler.transform(trainDf[col])
    testDf[col] = scaler.transform(testDf[col])

trainDf["rot45_X"] = .707 * trainDf["Y"] + .707 * trainDf["X"]
trainDf["rot45_Y"] = .707 * trainDf["Y"] - .707 * trainDf["X"]
trainDf["rot30_X"] = (1.732 / 2) * trainDf["X"] + (1. / 2) * trainDf["Y"]
trainDf["rot30_Y"] = (1.732 / 2) * trainDf["Y"] - (1. / 2) * trainDf["X"]
trainDf["rot60_X"] = (1. / 2) * trainDf["X"] + (1.732 / 2) * trainDf["Y"]
trainDf["rot60_Y"] = (1. / 2) * trainDf["Y"] - (1.732 / 2) * trainDf["X"]
trainDf["radial_r"] = np.sqrt(np.power(trainDf["Y"], 2) + np.power(trainDf["X"], 2))

testDf["rot45_X"] = .707 * testDf["Y"] + .707 * testDf["X"]
testDf["rot45_Y"] = .707 * testDf["Y"] - .707 * testDf["X"]
testDf["rot30_X"] = (1.732 / 2) * testDf["X"] + (1. / 2) * testDf["Y"]
testDf["rot30_Y"] = (1.732 / 2) * testDf["Y"] - (1. / 2) * testDf["X"]
testDf["rot60_X"] = (1. / 2) * testDf["X"] + (1.732 / 2) * testDf["Y"]
testDf["rot60_Y"] = (1. / 2) * testDf["Y"] - (1.732 / 2) * testDf["X"]
testDf["radial_r"] = np.sqrt(np.power(testDf["Y"], 2) + np.power(testDf["X"], 2))

# Attributes used in the model
print list(trainDf.columns.values)
print list(testDf.columns.values)

# back to numpy format
trainData = trainDf.values
testData = testDf.values

plt.boxplot(trainData)
plt.show()

# Feature Selection:
# The Recursive Feature Elimination (RFE) method is a feature selection approach. It works by recursively removing
# attributes and building a model on those attributes that remain. It uses the model accuracy to identify which
# attributes (and combination of attributes) contribute the most to predicting the target attribute.
'''
model = LogisticRegression()
# create the RFE model and select n attributes
rfe = RFE(model, 5)
rfe = rfe.fit(trainData[0::, 1::], trainData[0::, 0])
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

trainData = SelectKBest(f_classif, k=3).fit_transform(trainData[0::,1::], trainData[0::,0])
print 'F Values: ', f_classif(trainData[0::,1::], trainData[0::,0])[0]
print 'P Values: ', f_classif(trainData[0::,1::], trainData[0::,0])[1]
'''
print 'Training...'
forest_v = RandomForestClassifier(max_depth=16,n_estimators=256, oob_score=True)
forest = forest_v.fit(trainData[0::,1::], trainData[0::,0])

# Feature importances
#importances = forest.feature_importances_
#print "Feature Importances: ", importances

print 'Predicting...'
output = forest.predict_proba(testData).astype(float)

output = output.tolist()
predictions_file = open("../submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id",'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
                           'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION',
                           'FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT',
                           'LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
                           'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY',
                           'SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY',
                           'SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
                           'WEAPON LAWS'])
for x in range(len(output)):
    output[x].insert(0, x)
    open_file_object.writerow(output[x])
predictions_file.close()
print 'Done.'

print 'Plotting Learning Curves...'
auxiliary.plotLearningCurves(trainDf, forest)

