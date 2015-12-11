# https://www.kaggle.com/wendykan/sf-crime/don-t-know-what-i-want-to-do-yet
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv', header=0, parse_dates=['Dates'])
test = pd.read_csv('../data/test.csv', header=0, parse_dates=['Dates'])

train['Year'] = train['Dates'].map(lambda x: x.year)
train['Week'] = train['Dates'].map(lambda x: x.week)
train['Hour'] = train['Dates'].map(lambda x: x.hour)

train['event'] = 1
hourly_district_events = train[['PdDistrict', 'Hour', 'event']].groupby(['PdDistrict', 'Hour']).count().reset_index()
hourly_district_events_pivot = hourly_district_events.pivot(index='Hour', columns='PdDistrict',
                                                            values='event').fillna(method='ffill')
hourly_district_events_pivot.interpolate().plot(title='number of cases hourly by district', figsize=(10,6))

plt.show()
#plt.savefig('hourly_events_by_district.png')
