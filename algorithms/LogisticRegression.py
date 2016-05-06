__author__ = 'kunal'

import csv
from sklearn.linear_model import LogisticRegression as lr
import auxiliary

trainDf = auxiliary.initialise_train(False)

# select all columns except
# Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y,Year,Week,Hour
trainDf = trainDf.drop(['Dates', 'Descript', 'DayOfWeek', 'Resolution', 'Address'], axis=1)

# Test data
testDf = auxiliary.initialise_test(False)
ids = testDf['Id'].values
# Id,Dates,DayOfWeek,PdDistrict,Address,X,Y,Year,Week,Hour
testDf = testDf.drop(['Id', 'Dates', 'Address', 'DayOfWeek'], axis=1)

# Random Forest Algorithm
print list(trainDf.columns.values)
print list(testDf.columns.values)
# print list(trainDf.X.values)

# back to numpy format
trainData = trainDf.values
testData = testDf.values

print 'Training...'
logit = lr()
logit = logit.fit(trainData[0::, 1::], trainData[0::, 0])

print 'Predicting...'
output = logit.predict_proba(testData).astype(float)
output = output.tolist()

predictions_file = open("../submissionLR.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id", 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
                           'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
                           'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING',
                           'LARCENY/THEFT',
                           'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
                           'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
                           'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
                           'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS',
                           'WEAPON LAWS'])
for x in range(len(output)):
    output[x].insert(0, x)
    open_file_object.writerow(output[x])
predictions_file.close()
print 'Done.'
