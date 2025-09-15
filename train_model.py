import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("car_cleaned_data.csv")

# Example: Suppose 'Price' is the target variable
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to file
pickle.dump(model, open("model.pkl", "wb"))
