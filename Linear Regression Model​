import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("city_day.csv")

# 2. Select important columns
df = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']]

# 3. Handle missing values
df = df.dropna()

# 4. Split features and target
X = df.drop('AQI', axis=1)
y = df['AQI']

# 5. Train-test split (IMPORTANT: use random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 9. Print results
print("Linear Regression Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
