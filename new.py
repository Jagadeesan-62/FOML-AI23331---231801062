import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # For saving the model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('cozywinterdata.csv')  # Replace with your actual sales dataset path

# Step 1: Data Preprocessing
data.dropna(inplace=True)  # Drop rows with missing values

# Convert Date to datetime object (let pandas infer the date format)
data['Date'] = pd.to_datetime(data['Date'])

# Clean non-numeric characters and convert to numeric (Price, Competitor Price, etc.)
def clean_numeric_column(col):
    return col.replace({'\$': '', ',': ''}, regex=True).astype(float)

# Clean the relevant columns
data['Price'] = clean_numeric_column(data['Price'])
data['Competitor Price'] = clean_numeric_column(data['Competitor Price'])
data['Sales Revenue'] = clean_numeric_column(data['Sales Revenue'])

# Encode the 'Promotion' column: 'Yes' -> 1, 'No' -> 0
label_encoder = LabelEncoder()
data['Promotion'] = label_encoder.fit_transform(data['Promotion'])

# One-Hot Encoding for other Categorical Variables
# Encoding categorical columns (e.g., 'Product Name', 'Category', 'Weather', 'Holiday Season', etc.)
data = pd.get_dummies(data, columns=['Product Name', 'Category', 'Weather', 'Holiday Season', 'Event Impact', 'Seasonality'], drop_first=True)

# Normalize/Standardize Numerical Features
scaler = StandardScaler()
numerical_features = ['Sales Units', 'Stock Level', 'Price', 'Economic Indicator', 'Competitor Price', 'Website Traffic', 'Lead Time Days', 'Returns', 'Demand Forecast', 'Promotion']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Feature and Target Separation
X = data.drop(columns=['Sales Revenue', 'Date'])  # Drop 'Sales Revenue' (target) and 'Date' column (if not used for time series)
y = data['Sales Revenue']

# Step 2: Split Dataset
train_test_data = X.iloc[:-50]  # First 9950 rows for training/testing
train_test_target = y.iloc[:-50]
input_data = X.iloc[-50:]  # Last 50 rows for prediction
input_target = y.iloc[-50:]

X_train, X_test, y_train, y_test = train_test_split(train_test_data, train_test_target, test_size=0.2, random_state=42)

# Step 3: Model Definition
# XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

# Ensemble Model using Stacking Regressor (XGBoost + Random Forest)
ensemble_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    final_estimator=LinearRegression()
)
ensemble_model.fit(X_train, y_train)

# Save Models
joblib.dump(ensemble_model, 'ensemble_model_sales_revenue.pkl')
joblib.dump(scaler, 'scaler_sales_revenue.pkl')

# Step 4: Evaluate the Model
y_pred_test = ensemble_model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 5: Residual Analysis
residuals = y_test - y_pred_test

# Residuals Histogram
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Residuals vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5, color='orange')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Sales Revenue')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sales Revenue')
plt.show()

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Matrix')
plt.show()

# Step 7: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Sales Revenue')
plt.ylabel('Predicted Sales Revenue')
plt.title('Actual vs Predicted Sales Revenue')
plt.legend()
plt.show()

# Step 8: Predict on Input Data
input_data_scaled = scaler.transform(input_data)
input_predictions = ensemble_model.predict(input_data_scaled)

# Save Input Data and Predictions
input_data.to_csv('input_data_sales_revenue.csv', index=False)
pd.DataFrame(input_predictions, columns=['Predicted_Sales_Revenue']).to_csv('predicted_sales_revenue.csv', index=False)
