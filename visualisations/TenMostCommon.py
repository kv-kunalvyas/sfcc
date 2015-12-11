# https://www.kaggle.com/lllinger/sf-crime/crimedistribution-temporal-spatial/files
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv', header=0)

ten_most_common = train[train['Category'].isin(train['Category'].value_counts().head(10).index)]

ten_most_crime_by_district = pd.crosstab(ten_most_common['PdDistrict'], ten_most_common['Category'])
ten_most_crime_by_district.plot(kind='barh', figsize=(16,10), stacked=True, colormap='Greens',
                                title='Distribution of the City-wide Ten Most Common Crimes in Each District')
plt.show()
# plt.savefig('Disbribution_of_the_City-wide_Ten_Most_Common_Crimes_in_Each_District.png')

