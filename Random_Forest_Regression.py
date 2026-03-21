import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the Dataframe and print the first five rows from the dataset
Data_Frame = pd.read_csv('city_day.csv')
print("First 5 rows of city_day.csv: ")
print(Data_Frame.head())

# Drop all rows that have the target variable (AQI) missing
Data_Frame = Data_Frame.dropna(subset=["AQI"])

# Drop all non-useful columns
Data_Frame = Data_Frame.drop(columns=["City", "Date","AQI_Bucket"], errors = "ignore")

# Fill missing values with column mean
Data_Frame = Data_Frame.fillna(Data_Frame.mean(numeric_only=True))

# Seperate features and target
X = Data_Frame.drop("AQI", axis = 1)
y = Data_Frame["AQI"]

# Train Test Split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)