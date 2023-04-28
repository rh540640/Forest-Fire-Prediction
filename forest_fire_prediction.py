# -*- coding: utf-8 -*-
"""forest-fire-prediction.ipynb

# Forest Fire Prediction

> Richard H
"""

# import libraries
import pandas as pd
import numpy as np

# load the data
path = '../input/forest-fires-data-set/forestfires.csv'
forestfires = pd.read_csv(path)

forestfires.head()

forestfires.info()

forestfires['month'].value_counts()

forestfires['day'].value_counts()

forestfires.describe()

"""   1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
   2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
   3. month - month of the year: "jan" to "dec" 
   4. day - day of the week: "mon" to "sun"
   5. FFMC - FFMC index from the FWI system: 18.7 to 96.20; Fine fuel moisture code representing the moisture content of litter
   6. DMC - DMC index from the FWI system: 1.1 to 291.3; Duff moisture code representing the average moisture content of organic layers and woody material
   7. DC - DC index from the FWI system: 7.9 to 860.6; Drought moisture code representing the average moisture content of organic layers
   8. ISI - ISI index from the FWI system: 0.0 to 56.10
   9. temp - temperature in Celsius degrees: 2.2 to 33.30
   10. RH - relative humidity in %: 15.0 to 100
   11. wind - wind speed in km/h: 0.40 to 9.40 
   12. rain - outside rain in mm/m2 : 0.0 to 6.4 
   13. area - the burned area of the forest (in hectares (ha)): 0.00 to 1090.84

"""

import matplotlib.pyplot as plt

forestfires.hist(bins=50, figsize=(12, 8))
plt.show()

forestfires['area'].hist(bins=50)
plt.xlabel('area burned in ha')
plt.ylabel('count')
plt.show()

"""The data is heavily skewed towards 0.0

# Create a train and test set
"""

# put data into 3 categories based on area burned
forestfires['area_categories'] = pd.cut(forestfires['area'], bins=[-1.0, 0.52, 6.57, 1090.84])

forestfires['area_categories'].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.ylabel('Count')
plt.xlabel('area burned (ha)')
plt.show()

from sklearn.model_selection import train_test_split

# Create a stratified train and test set based on area category
strat_train_set, strat_test_set = train_test_split(forestfires, test_size=0.2, stratify=forestfires['area_categories'], random_state=42)

# Compare ratio of 'area_categories'
forestfires['area_categories'].value_counts() / len(forestfires)

strat_test_set['area_categories'].value_counts() / len(strat_test_set)

"""# Analysis"""

forestfires = strat_train_set.copy()

forestfires['log_area'] = np.log(forestfires['area'] + 1)

# Create new column 'burned'. 0 if area = 0.0, else 1
forestfires['burned'] = forestfires['area'].apply(lambda x: 0.0 if x == 0.0 else 1.0)

# Get all forest fires where burned is 1
forestfires_burned = forestfires[forestfires['burned'] != 0.0]

# Group the forestfires by their X and Y coordinate, get the count of how many are in each point, and reset the index so that we can plot it by the X and Y
forestfires_burned_count = forestfires_burned.groupby(['X', 'Y']).size().reset_index()
forestfires_burned_count.plot(kind='scatter', x='X', y='Y', grid=True, c=0, cmap='jet', colorbar=True, legend=True)
plt.show()

"""The graph shows that row four has more fires that have more than 0 hectares burnt than other points. Also (6,5) and (8,6) have around 20 fires that have occured."""

forestfires = forestfires.drop('burned', axis=1)

correlation = forestfires.corr()
correlation['area'].sort_values(ascending=False)

# Create scatterplots comparing important columns with area
columns = ['X', 'Y','FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
for column in columns:
    plt.scatter(forestfires[column], forestfires['area'])
    plt.title(column + ' compared to burned area')
    plt.xlabel(column)
    plt.ylabel('area')
    plt.show()

# do the same with the 'log_area'
for column in columns:
    plt.scatter(forestfires[column], forestfires['log_area'])
    plt.title(column + ' compared to burned area')
    plt.xlabel(column)
    plt.ylabel('area')
    plt.show()

correlation = forestfires.corr()
correlation['log_area'].sort_values(ascending=False)

# find the average area burned for each day
days = forestfires['day'].unique()
avg_area_days = []
for day in days:
    avg_area_days.append(forestfires[forestfires['day'] == day]['area'].mean())

plt.bar(days, avg_area_days)
plt.title('Days compared to the average area burned')
plt.xlabel('Days')
plt.ylabel('Average area burned (ha)')
plt.show()

forestfires['month'].value_counts()

months = list(forestfires['month'].unique())

# drop the values that occured in the months with few values
drop_months = ['may', 'jan', 'dec', 'apr']
forestfires = forestfires[forestfires['month'].isin(drop_months) == False]

# find the average area burned for each month
months = list(forestfires['month'].unique())  
avg_area_month = []
for month in months:
    avg_area_month.append(forestfires[forestfires['month'] == month]['area'].mean())

plt.bar(months, avg_area_month)
plt.title('Months compared to the average area burned')
plt.xlabel('Months')
plt.ylabel('Average area burned (ha)')
plt.show()

"""# Preprocessing"""

forestfires = strat_train_set
forestfires['area'] = np.log(forestfires['area'] + 1)
strat_test_set['area'] = np.log(strat_test_set['area'] + 1)
forestfires.drop(columns='area_categories', inplace=True)

display(forestfires)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# create a pipeline that uses OneHotEncoder to transform the cat_columns and MinMaxScaler on the num_columns
num_columns = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
cat_columns = ['month', 'day']
preprocessing = ColumnTransformer([
    ('cat', OneHotEncoder(), cat_columns),
    ('minmax', MinMaxScaler(), num_columns)
])

"""# Train the models"""

X_train = preprocessing.fit_transform(forestfires.drop(columns='area'))
y_train = forestfires['area']

X_test = preprocessing.fit_transform(strat_test_set.drop(columns='area'))
y_test = strat_test_set['area']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(mse)

orig_mse = np.exp(mse) - 1
print(orig_mse)

predictions