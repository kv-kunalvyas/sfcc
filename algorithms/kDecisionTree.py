__author__ = 'kunal'

import csv
from sklearn import tree
import auxiliary
import pylab as P
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
import numpy as np
import warnings

warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', Warning)

trainDf = auxiliary.initialise_train(False)
# auxiliary.computeMean(Category)
# select all columns except
trainDf = trainDf.drop(['Dates', 'Descript', 'Resolution', 'Address'], axis=1)

# Test data
testDf = auxiliary.initialise_test(False)
ids = testDf['Id'].values
testDf = testDf.drop(['Id', 'Dates', 'Address'], axis=1)

# Attributes used in the model
print list(trainDf.columns.values)
print list(testDf.columns.values)

# back to numpy format
trainData = trainDf.values
testData = testDf.values

print 'Training...'
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(trainData[0::,1::], trainData[0::,0])

print 'Predicting...'
output = dtree.predict_proba(testData).astype(float)
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

# TODO: Plot ROC curves
P.show()

# assume classifier and training data is prepared...
features_list = trainDf.columns.values[1::]
X = trainDf.values[:, 1::]
y = trainDf.values[:, 0]

train_sizes, train_scores, test_scores = learning_curve(
        dtree, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Decision Tree")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
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
dtree.fit(X_train, y_train)
plt.show()