import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Load dataset
    dataset = pd.read_csv("house_price_dataset.csv")

    print("\nDataset Loaded Successfully!")
    print(dataset.head())
    print("\nDataset Shape:", dataset.shape)

    # Split Dataset into Features and Target
    X = dataset[["Area", "Bedrooms", "Age", "DistanceCity"]]  # Features
    y = dataset["Price"]                                      # Target

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Data Size:", X_train.shape)
    print("Testing Data Size:", X_test.shape)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel training completed!")

    # Check model coefficients
    print("\nIntercept:", model.intercept_)
    print("\nCoefficients:")

    for feature, coef in zip(X.columns, model.coef_):
        print(feature, ":", coef)

    # Predict on Test Data
    y_pred = model.predict(X_test)
    print("\nPrediction completed!")

    # Evaluate Model Performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Results:")
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R² Score:", r2)

    # Test with Custom Input
    new_house = pd.DataFrame({
        "Area": [2000],
        "Bedrooms": [3],
        "Age": [10],
        "DistanceCity": [5]
    })

    predicted_price = model.predict(new_house)

    print("\nCustom Test Case:")
    print(new_house)
    print("Predicted Price:", predicted_price[0])

    # Compare Actual vs Predicted (First 10 values)
    comparison = pd.DataFrame({
        "Actual Price": y_test.values[:10],
        "Predicted Price": y_pred[:10]
    })

    print("\nActual vs Predicted (First 10 Results):")
    print(comparison)


if __name__ == "__main__":
    main()
