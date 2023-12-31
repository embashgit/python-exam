import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


class LeastSquaresRegression:
    def fit(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

def calculate_least_squares(X, y):
    model = LeastSquaresRegression()

    # Fit the model to the data
    model_fit = model.fit(X, y)

    # Predict Y values using the fitted model
    y_pred = model_fit.predict(X)

    # Calculate the mean squared error
    mse = mean_squared_error(y, y_pred)

    return model_fit, mse

def main():
    training_data = pd.read_csv('train.csv')

    # Assuming X column in the training data
    X_train = training_data[['x']]

    # List of Y columns in the training data
    y_columns = ['y1', 'y2', 'y3', 'y4']

    # Dictionary to store results for each Y column
    results = {}

    # Calculate Least-Squares Regression for each Y column
    for y_col in y_columns:
        Y_train = training_data[y_col]

        # Calculate Least-Squares Regression for the current Y column
        model_fit, mse = calculate_least_squares(X_train, Y_train)

        # Store the results in the dictionary
        results[y_col] = {'Coefficients': model_fit.coef_[0],
                          'Intercept': model_fit.intercept_,
                          'Mean Squared Error': mse}

    # Print the results
    for y_col, result in results.items():
        print(f"Results for {y_col}:")
        print("Coefficients:", result['Coefficients'])
        print("Intercept:", result['Intercept'])
        print("Mean Squared Error:", result['Mean Squared Error'])
        print("\n")

if __name__ == "__main__":
    main()