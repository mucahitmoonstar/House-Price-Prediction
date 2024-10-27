import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load the dataset using sklearn.datasets
housing = fetch_california_housing()
X = housing.data  # Features
y = housing.target  # Target variable (house prices)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Perform feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Use a regression model like Linear Regression, Random Forest, or Gradient Boosting
# Option 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Option 2: Random Forest Regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)

# Option 3: Gradient Boosting Regressor (Potentially better performance)
gbr_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
gbr_model.fit(X_train_scaled, y_train)

# Predict house prices using the trained models
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = random_forest_model.predict(X_test_scaled)
y_pred_gbr = gbr_model.predict(X_test_scaled)

# 4. Visualize the predicted vs actual house prices using matplotlib
plt.figure(figsize=(18, 6))



# Visualization for Random Forest Regressor
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_rf, color='green', edgecolors='k', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Random Forest - Predicted vs Actual House Prices')
plt.grid(True)



plt.tight_layout()
plt.show()

# Print model performance for all models

print('Random Forest Model Performance:')
print(f'Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_rf):.2f}')
print(f'R-squared Score: {r2_score(y_test, y_pred_rf):.2f}')
print(f'Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_rf):.2f}\n')

