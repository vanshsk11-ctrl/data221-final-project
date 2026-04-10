# Air Pollution Prediction using Machine Learning

## Project Overview
This project focuses on predicting Air Quality Index (AQI) values in India using machine learning models. The dataset contains daily air pollution measurements from multiple cities across India between 2015 and 2020. The goal is to compare different models and evaluate their performance using consistent methods and metrics.

## Dataset
We used the “Air Quality Data in India (2015–2020)” dataset from Kaggle. Each row represents pollution measurements for a specific city on a given day. Features include pollutants such as PM2.5, PM10, NO2, CO, SO2, and O3, and the target variable is AQI.

## Models Used

### Linear Regression
A simple baseline model that assumes a linear relationship between pollutants and AQI. It is easy to interpret but may not capture complex patterns in the data.

### Decision Tree Regression
A nonlinear model that splits the data based on feature values. It can capture more complex relationships but may overfit if not controlled.

### Random Forest Regression
An ensemble model that combines multiple decision trees. It reduces overfitting and generally provides strong performance for tabular data.

### Support Vector Regression (SVR)
A nonlinear model that uses the RBF kernel to capture complex relationships. It requires feature scaling and parameter tuning but can perform well on structured data.

## Evaluation Metrics
All models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Cross-validation was also used to assess model stability.

## Conclusion
The models were compared based on their performance metrics. Nonlinear models generally performed better, with Random Forest achieving the best results, while SVR also showed strong performance after tuning.
