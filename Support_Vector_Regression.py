import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Fill missing values with column mean
Data_Frame = Data_Frame.fillna(Data_Frame.mean(numeric_only=True))

# Separate features and target
X = Data_Frame.drop("AQI", axis=1)
y = Data_Frame["AQI"]

# Train Test Split, 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# Test different SVR parameters
print("\nTesting different SVR parameters:")

param_options = [
    {"C": 10, "gamma": 0.1},
    {"C": 100, "gamma": 0.1},
    {"C": 100, "gamma": 0.01}
]

for params in param_options:
    temp_model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, temp_pred))
    print(f"C={params['C']}, gamma={params['gamma']} -> RMSE: {rmse}")


# Compare actual vs predicted values
results_df = pd.DataFrame({
    "Actual AQI": y_test.values,
    "Predicted AQI": y_pred
})

print("\nSample predictions:")
print(results_df.head())