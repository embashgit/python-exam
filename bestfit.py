from scipy.optimize import curve_fit
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# Define the 50 ideal functions
def f1(x, a, b, c):
    return a * np.sin(b * x) + c

def f2(x, a, b, c):
    return a * np.cos(b * x) + c

def f3(x, a, b, c):
    return a * np.exp(b * x) + c

# Define the function that chooses the four ideal functions
def choose_ideal_functions(training_data, ideal_functions):
    ideal_function_params = []
    for i in range(4):
        x_data = training_data[i][0]
        y_data = training_data[i][1]
        best_fit_function = None
        best_fit_params = None
        best_fit_error = float('inf')
        for j in range(len(ideal_functions)):
            try:
                params, _ = curve_fit(ideal_functions[j], x_data, y_data)
                y_fit = ideal_functions
                error = np.sum((y_data - y_fit) ** 2)
                if error < best_fit_error:
                    best_fit_function = ideal_functions[j]
                    best_fit_params = params
                    best_fit_error = error
            except:
                pass
        ideal_function_params.append((best_fit_function, best_fit_params))
    return ideal_function_params

# Define the function that maps the x-y-pairs of values to the four chosen ideal functions
def map_to_ideal_functions(x, ideal_function_params):
    y = []
    for i in range(len(x)):
        x_i = x[i]
        y_i = []
        for j in range(4):
            ideal_function = ideal_function_params[j][0]
            ideal_function_params_j = ideal_function_params[j][1]
            try:
                y_i_j = ideal_function(x_i, *ideal_function_params_j)
            except:
                y_i_j = None
                raise
            y_i.append(y_i_j)
        y.append(y_i)
    return y


# Define the training data
training_data = pd.read_csv('train.csv')

# Define the test data
test_data = pd.read_csv('test.csv')

# Choose the four ideal functions
ideal_functions = pd.read_csv('ideal.csv')
ideal_function_params = choose_ideal_functions(training_data, ideal_functions)

# Map the x-y-pairs of values to the four chosen ideal functions
x = test_data[0]
y = test_data[1]
y_mapped = map_to_ideal_functions(x, ideal_function_params)

# Print the results
print('Ideal functions:')
for i in range(4):
    print(f'Function {i + 1}: {ideal_function_params[i][0].__name__} with parameters {ideal_function_params[i][1]}')
print('Mapping of test data to ideal functions:')
for i in range(len(x)):
    print(f'x = {x[i]}, y = {y[i]}, y_mapped = {y_mapped[i]}')
