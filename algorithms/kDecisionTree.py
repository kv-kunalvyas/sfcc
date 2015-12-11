__author__ = 'kunal'

import csv
from sklearn import tree
from sklearn import metrics
import auxiliary
import warnings
from sklearn.cross_validation import train_test_split

warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', Warning)

# Train Data
trainDf = auxiliary.initialise_train(False)
sec_trainDf = auxiliary.initialise()
trainDf = trainDf.drop(['Dates', 'Descript', 'DayOfWeek', 'Resolution', 'Address'], axis=1)
X, y = train_test_split(trainDf, train_size=.75)

# Test data
testDf = auxiliary.initialise_test(False)
ids = testDf['Id'].values
testDf = testDf.drop(['Id', 'Dates', 'DayOfWeek', 'Address'], axis=1)
actual = y[0::,0].tolist()

# back to numpy format
trainData = trainDf.values
testData = testDf.values

print 'Training...'
dtree_v = tree.DecisionTreeClassifier()
dtree = dtree_v.fit(trainData[0::,1::], trainData[0::,0])
dtree_val = dtree_v.fit(X[0::,1::], X[0::,0])

importances = dtree.feature_importances_
print "Importances: ", importances

print 'Predicting...'
output = dtree.predict_proba(testData).astype(float)
predicted = dtree_val.predict_proba(y[0::,1::]).astype(float)

print "Calculating Multi Class Log Loss..."
print "Multi Class Log Loss on Validation set: ", metrics.log_loss(actual, predicted)

output = output.tolist()
predictions_file = open("../submissionDT.csv", "wb")
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
print 'Done!'

print "Plotting learning curves..."
auxiliary.plotLearningCurves(trainDf, dtree)