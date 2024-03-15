# -*- coding: utf-8 -*-
"""Assignment.ipynb

Original file is located at
    https://colab.research.google.com/drive/1t1HEEugjl6Hm41D4PeFyGVMfXaC9EKEo
"""

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
file_path = '/content/drive/MyDrive/new_vehicles_registered.csv'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data_csv = pd.read_csv(file_path, header=2)

# Clean and prepare the data
for col in data_csv.columns[2:]:
    data_csv[col] = pd.to_numeric(data_csv[col].str.replace(',', '').replace('-', np.nan), errors='coerce')

data_csv['CLASS OF VEHICLE'].fillna(method='ffill', inplace=True)
data_csv = data_csv.drop(0)
data_csv.columns = ['CLASS OF VEHICLE', 'Fuel Type', '2008', '2009', '2010', '2011', '2012']

yearly_totals = data_csv.groupby('Fuel Type')[['2008', '2009', '2010', '2011', '2012']].sum()
yearly_totals


# Sum the registrations across all years for each fuel type
total_registrations = data_csv[['2008', '2009', '2010', '2011', '2012']].sum()

# Create a pie chart for the summed data
plt.figure(figsize=(8, 8))
plt.pie(total_registrations, labels=total_registrations.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Vehicle Registrations by Year')
plt.show()

total_registrations = data_csv.groupby('Fuel Type').sum().sum(axis=1)

# Plotting the pie chart for the sum of all years
plt.figure(figsize=(8, 8))
plt.pie(total_registrations, labels=total_registrations.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Vehicle Registrations by Fuel Type (2008-2012)')
plt.show()


# Plot the total number of vehicles registered per year for each fuel type
fig, ax = plt.subplots(figsize=(10, 6))
yearly_totals.T.plot(kind='bar', ax=ax)
ax.set_title('Total Number of Vehicles Registered by Fuel Type (2008-2012)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Vehicles Registered')
ax.legend(title='Fuel Type')
plt.tight_layout()
plt.show()

# Calculate the year-over-year change rate for each fuel type
change_rates = yearly_totals.pct_change(axis='columns') * 100

# Plot the year-over-year change rates for each fuel type
fig, ax = plt.subplots(figsize=(10, 6))
change_rates.T.plot(kind='line', ax=ax, marker='o')
ax.set_title('Year-Over-Year Change Rate in Vehicle Registrations by Fuel Type')
ax.set_xlabel('Year')
ax.set_ylabel('Change Rate (%)')
ax.legend(title='Fuel Type')
ax.grid(True)
plt.tight_layout()
plt.show()


# Prepare the data for the linear regression model
X = np.array([2008, 2009, 2010, 2011, 2012]).reshape(-1, 1)
predictions = {}

# Create a linear regression model for each fuel type
for fuel_type in yearly_totals.index:
    y = yearly_totals.loc[fuel_type].values
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(np.array([[2013]]))
    predictions[fuel_type] = int(prediction[0])

predictions_df = pd.DataFrame(list(predictions.items()), columns=['Fuel Type', 'Predicted 2013 Registrations'])
predictions_df.set_index('Fuel Type', inplace=True)
predictions_df
