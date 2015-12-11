__author__ = 'kunal'
# best score on test 2.58420
import csv
from sklearn.ensemble import RandomForestClassifier as rfc
import auxiliary
import warnings
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.feature_selection import f_classif, SelectKBest
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', Warning)

trainDf = auxiliary.initialise_train(False)

trainDf = trainDf.drop(['Dates', 'Descript', 'DayOfWeek', 'Resolution', 'Address'], axis=1)
X, y = train_test_split(trainDf, train_size=.75)

# Test data
testDf = auxiliary.initialise_test(False)
ids = testDf['Id'].values
testDf = testDf.drop(['Id', 'Dates', 'Address', 'DayOfWeek'], axis=1)
actual = y[0::,0].tolist()

# back to numpy format
trainData = trainDf.values
testData = testDf.values

# Feature Selection:
'''
trainData = SelectKBest(f_classif, k=3).fit_transform(trainData[0::,1::], trainData[0::,0])
print 'F Values: ', f_classif(trainData[0::,1::], trainData[0::,0])[0]
print 'P Values: ', f_classif(trainData[0::,1::], trainData[0::,0])[1]
'''
print 'Training...'
# Deciding best parameters for Random Forest
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
best_cv_score = -9999.9999
best_n_est = 10000
avg_scores = []
for i in n_estimators:
    forest = rfc(n_estimators=i, oob_score=True)
    scores = cross_val_score(forest, trainData[0::,1::], trainData[0::,0], scoring='log_loss', cv=5, n_jobs=-1)
    avg_scores.append(auxiliary.calc_avg(scores))
    if avg_scores[-1] > best_cv_score:
        best_cv_score = avg_scores[-1]
        best_n_est = i
plt.plot(n_estimators, avg_scores)
plt.show()

forest_v = rfc(n_estimators=best_n_est, oob_score=True)
forest = forest_v.fit(trainData[0::,1::], trainData[0::,0])
forest_val = forest_v.fit(X[0::,1::], X[0::,0])

print 'Features: ', list(testDf.columns.values)
importances = forest.feature_importances_
print "Feature Importances: ", importances

print 'Predicting...'
output = forest.predict_proba(testData).astype(float)
predicted = forest_val.predict_proba(y[0::,1::]).astype(float)

print "Multi Class Log Loss on Validation set: ", metrics.log_loss(actual, predicted)

output = output.tolist()
predictions_file = open("../submissionRF.csv", "wb")
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