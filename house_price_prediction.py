import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv("dataset/Housing.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())


# Data visualization

sns.set_style("whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="bedrooms", y="price")
plt.title("Bedrooms vs Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="bathrooms", y="price")
plt.title("Bathrooms vs Price")
plt.xlabel("Bathrooms")
plt.ylabel("Price")
plt.show()


# Selecting input and output columns

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]


# Splitting the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Training the model

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# Predicting house prices

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.astype(int)
})

print("\nSample Predictions")
print(results.head(10))


# Evaluating the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 35)
print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")


# Comparing actual and predicted prices

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()


# Predict price for a new house

print("\nHouse Price Prediction")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms]
})

predicted_price = model.predict(house)

print(f"\nEstimated House Price: ₹ {predicted_price[0]:,.2f}")