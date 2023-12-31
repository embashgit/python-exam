import numpy as np
from sklearn.linear_model import LinearRegression

class LeastSquaresRegression:
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def fit(self, degree):
        # Validate input data
        if len(self.x_values) != len(self.y_values):
            raise ValueError("Length of x_values must match the length of y_values.")

        coefficients = np.polyfit(self.x_values, self.y_values, degree)
        return coefficients


class LeastSquaresRegressionAlt:
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        self.model = LinearRegression()
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)