__author__ = 'kunal'
import csv
from sklearn.naive_bayes import GaussianNB
import auxiliary
from sklearn import metrics

trainDf = auxiliary.initialise_train(False)
# auxiliary.computeMean(Category)

# select all columns except
trainDf = trainDf.drop(['Dates', 'Descript', 'DayOfWeek', 'Resolution', 'Address'], axis=1)

# Test data
testDf = auxiliary.initialise_test(False)
ids = testDf['Id'].values
testDf = testDf.drop(['Id', 'Dates', 'DayOfWeek', 'Address'], axis=1)

# Attributes used in the model
print list(trainDf.columns.values)
print list(testDf.columns.values)

# back to numpy format
trainData = trainDf.values
testData = testDf.values

print 'Training...'
bayes = GaussianNB()
bayes = bayes.fit(trainData[0::,1::], trainData[0::,0])

print 'Predicting...'
output = bayes.predict_proba(testData).astype(float)
output = output.tolist()

predictions_file = open("../submissionNB.csv", "wb")
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
