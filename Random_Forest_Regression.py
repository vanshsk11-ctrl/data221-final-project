import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Tuning the model
parameter_grid ={
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

Random_Forest_Model = RandomForestRegressor(random_state = 42)

grid_search = GridSearchCV(estimator=Random_Forest_Model, param_grid = parameter_grid, cv=5, scoring='neg_mean_squared_error',n_jobs=-1)

print("\n Training Random Forest Model")
grid_search.fit(X_train, y_train)

print("\n Best Parameters: ")
print(grid_search.best_params_)

# Train the Best Model
Best_Random_Forest_Model = grid_search.best_estimator_

# Predictions
y_predictions = Best_Random_Forest_Model.predict(X_test)

#Evaluation
Mean_Absolute_Error = mean_absolute_error(y_test, y_predictions)
Root_Mean_squared_Error = np.sqrt(mean_squared_error(y_test, y_predictions))
r2 = r2_score(y_test, y_predictions)

print("\n RANDOM FOREST MODEL RESULTS ")
print(f"Mean Absolute Error:  {Mean_Absolute_Error}")
print(f"Root Mean Squared Error:  {Root_Mean_squared_Error}")
print(f"R2 Score:  {r2}")

# Feature importances

importances = Best_Random_Forest_Model.feature_importances_

feature_importance = pd.DataFrame({"Feature": X.columns,"Importances": importances,}).sort_values(by='Importances', ascending=False)

print("\n Feature importances: ")
print(feature_importance)

