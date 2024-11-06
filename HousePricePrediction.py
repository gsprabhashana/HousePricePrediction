import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('housing.csv')
df.head()  # Display the first few rows

# Check for missing values
print(df.isnull().sum())

# Fill missing values with median for numeric columns only
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
df = pd.read_csv('housing.csv')
df.head()  # Display the first few rows

# Check for missing values
print(df.isnull().sum())

# Fill missing values with median for numeric columns only
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Example: Boxplot for detecting outliers in the 'price' column
# Check if 'price' is in the columns, if not, print available columns
if 'price' in df.columns:
    sns.boxplot(df['price'])
    plt.show()

    # Optionally, remove outliers beyond 3 standard deviations
    df = df[(np.abs(df['price'] - df['price'].mean()) <= (3 * df['price'].std()))]
else:
    print(f"'price' column not found. Available columns: {df.columns.tolist()}")

# Check for missing values
print(df.isnull().sum())

# Display summary statistics
print(df.describe())

# Assuming 'df' is your DataFrame
# Select only numerical features for correlation calculation
numerical_features = df.select_dtypes(include=np.number).columns

# Calculate correlation for numerical features
correlation_matrix = df[numerical_features].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Fill missing values in 'total_bedrooms' with the median
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Replace missing values in 'total_bedrooms' with the median
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Feature engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

# One-hot encoding for 'ocean_proximity'
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Standardize features
scaler = StandardScaler()
numeric_features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Define X and y
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions and evaluation
y_pred_lr = lr.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R^2:", r2_score(y_test, y_pred_lr))

# Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(X_train, y_train)

# Predictions and evaluation
y_pred_dtr = dtr.predict(X_test)
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred_dtr))
print("Decision Tree R^2:", r2_score(y_test, y_pred_dtr))

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rfr.fit(X_train, y_train)

# Predictions and evaluation
y_pred_rfr = rfr.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rfr))
print("Random Forest R^2:", r2_score(y_test, y_pred_rfr))

# Comparison plot for Actual vs Predicted for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rfr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted Values (Random Forest)")
plt.show()
