import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('sale.csv')

# Check for NaN values
print(df.isna().sum())

# Drop rows with NaN values in the target column
df = df.dropna(subset=['TotalPrice'])

# Define features and target based on available columns
features = ['Quantity', 'UnitPrice']  # Example feature columns
target = 'TotalPrice'  # Example target column

# Select the features and target
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Sklearn: Predicting future sales
new_data = pd.DataFrame({
    'Quantity': [10],
    'UnitPrice': [15]
})
# Predict future sales
future_sales_prediction = model.predict(new_data)
print(f'Predicted Future Sales: {future_sales_prediction}')


print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Total Price')
plt.ylabel('Predicted Total Price')
plt.title('Actual vs Predicted Total Price')
plt.legend()
plt.show()


