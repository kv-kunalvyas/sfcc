# https://www.kaggle.com/catherinekherian/sf-crime/sf-crimes-show-two-peaks-each-month
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv', header=0, parse_dates=['Dates'])
test = pd.read_csv('../data/test.csv', header=0, parse_dates=['Dates'])

#Add day of the year format 02-22
train['DayOfYear'] = train['Dates'].map(lambda x: x.strftime("%m-%d"))
test['DayOfYear'] = test['Dates'].map(lambda x: x.strftime("%m-%d"))

train_days = train[["X", "DayOfYear"]].groupby(['DayOfYear']).count().rename(columns={"X": "TrainCount"})
test_days = test[ ["X", "DayOfYear"]].groupby(['DayOfYear']).count().rename(columns={"X": "TestCount"})

days = train_days.merge(test_days, left_index=True, right_index=True)
days["TotalCount"] = days["TrainCount"] + days["TestCount"]

days.plot(figsize=(15,10))
plt.title("The two peaks per month pattern is entirely explained by splitting the data into train/test sets")
plt.ylabel('Number of crimes')
plt.xlabel('Day of year')
plt.grid(True)

plt.show()
#plt.savefig('Distribution_of_Crimes_by_DayofYear.png')