import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load the Dataframe and print the first five rows
Data_Frame = pd.read_csv('city_day.csv')
print("First 5 rows of city_day.csv:")
print(Data_Frame.head())

# Drop rows with missing target (AQI)
Data_Frame = Data_Frame.dropna(subset=["AQI"])

# Drop non-useful columns
Data_Frame = Data_Frame.drop(columns=["City", "Date", "AQI_Bucket"], errors="ignore")

# Separate features and target
X = Data_Frame.drop("AQI", axis=1)
y = Data_Frame["AQI"]

# Train Test Split, 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fill missing values with column mean (using training data only to prevent leakage)
train_means = X_train.mean(numeric_only=True)
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVR model
model = SVR(kernel="rbf", C=100, gamma=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("SVR Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Test different SVR parameters
print("\nTesting different SVR parameters:")

param_grid = {
    "C": [10, 100],
    "gamma": [0.1, 0.01]
}

grid_search = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

for params, score in zip(
    grid_search.cv_results_["params"],
    grid_search.cv_results_["mean_test_score"]
):
    print(f"C={params['C']}, gamma={params['gamma']} -> RMSE: {-score:.4f}")

# Compare actual vs predicted values
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

results_df = pd.DataFrame({
    "Actual AQI": y_test.values,
    "Predicted AQI": y_pred
})

print("\nSample predictions:")
print(results_df.head())