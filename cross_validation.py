import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("city_day.csv")

# 2. Drop rows where target is missing
df = df.dropna(subset=["AQI"])

# 3. Drop unnecessary columns
df = df.drop(columns=["City", "Date", "AQI_Bucket"], errors="ignore")

# 4. Features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# 5. Define models
models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LinearRegression())
    ]),

    "Support Vector Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=100, gamma=0.1))
    ]),

    "Random Forest Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Decision Tree Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", DecisionTreeRegressor(
            random_state=42
        ))
    ])
}

# 6. Run 5-fold cross-validation for each model
results = []

for model_name, model in models.items():
    cv_mae = -cross_val_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    cv_rmse = np.sqrt(
        -cross_val_score(
            model, X, y,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
    )

    cv_r2 = cross_val_score(
        model, X, y,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    results.append({
        "Model": model_name,
        "Mean MAE": round(cv_mae.mean(), 2),
        "MAE Std": round(cv_mae.std(), 2),
        "Mean RMSE": round(cv_rmse.mean(), 2),
        "RMSE Std": round(cv_rmse.std(), 2),
        "Mean R2": round(cv_r2.mean(), 3),
        "R2 Std": round(cv_r2.std(), 3)
    })

# 7. Put results into table
results_df = pd.DataFrame(results)

# 8. Print table
print("\n5-Fold Cross-Validation Results:\n")
print(results_df.to_string(index=False))