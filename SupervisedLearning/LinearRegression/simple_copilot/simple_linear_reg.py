import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load CSV file
data = pd.read_csv("housing.csv")

# 2. Define features (X) and target (y)
X = data[["square_feet", "bedrooms"]]  # independent variables
y = data["price"]                      # dependent variable

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 7. Inspect coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
