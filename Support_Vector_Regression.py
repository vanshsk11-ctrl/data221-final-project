# Jasman Gill

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load the Dataframe and print the first five rows
dataframe = pd.read_csv('city_day.csv')
print("First 5 rows of city_day.csv:")
print(dataframe.head())

# Drop rows with missing target (AQI)
dataframe = dataframe.dropna(subset=["AQI"])

# Drop non-useful columns
dataframe = dataframe.drop(columns=["City", "Date", "AQI_Bucket"], errors="ignore")

# Separate features and target
features = dataframe.drop("AQI", axis=1)
target = dataframe["AQI"]

# Train Test Split, 80/20 split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Fill missing values with column mean (using training data only to prevent leakage)
training_means = features_train.mean(numeric_only=True)
features_train = features_train.fillna(training_means)
features_test = features_test.fillna(training_means)

# Scale features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Train SVR model
model = SVR(kernel="rbf", C=100, gamma=0.1)
model.fit(features_train, target_train)

# Make predictions
predictions = model.predict(features_test)

# Evaluate model
print("SVR Results:")
print("MAE:", mean_absolute_error(target_test, predictions))
print("RMSE:", root_mean_squared_error(target_test, predictions))
print("R2:", r2_score(target_test, predictions))

# Cross Validation (5-fold) - RMSE
cross_validation_rmse_scores = -cross_val_score(
    model,
    features_train,
    target_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

# Cross Validation (5-fold) - MAE
cross_validation_mae_scores = -cross_val_score(
    model,
    features_train,
    target_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

# Cross Validation (5-fold) - R2
cross_validation_r2_scores = cross_val_score(
    model,
    features_train,
    target_train,
    cv=5,
    scoring="r2"
)

print("\nCross Validation RMSE scores:", cross_validation_rmse_scores)
print("Average CV RMSE:", cross_validation_rmse_scores.mean())

print("\nCross-validation Scores:")
print("Mean:")
print("  MAE:", round(cross_validation_mae_scores.mean(), 2))
print("  RMSE:", round(cross_validation_rmse_scores.mean(), 2))
print("  R2:", round(cross_validation_r2_scores.mean(), 3))
print("Standard Deviation:")
print("  MAE:", round(cross_validation_mae_scores.std(), 2))
print("  RMSE:", round(cross_validation_rmse_scores.std(), 2))
print("  R2:", round(cross_validation_r2_scores.std(), 3))

# Test different SVR parameters
print("\nTesting different SVR parameters:")

parameter_grid = {
    "C": [10, 100],
    "gamma": [0.1, 0.01]
}

grid_search = GridSearchCV(
    SVR(kernel="rbf"),
    parameter_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(features_train, target_train)

for parameters, score in zip(
    grid_search.cv_results_["params"],
    grid_search.cv_results_["mean_test_score"]
):
    print(f"C={parameters['C']}, gamma={parameters['gamma']} -> RMSE: {-score:.4f}")

# Compare actual vs predicted values
best_model = grid_search.best_estimator_
predictions = best_model.predict(features_test)

results_dataframe = pd.DataFrame({
    "Actual AQI": target_test.values,
    "Predicted AQI": predictions
})

print("\nSample predictions:")
print(results_dataframe.head())