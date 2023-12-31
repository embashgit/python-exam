import numpy as np
from data_ingestion import DataIngestion
import pandas as pd
from least_square_regression import LeastSquaresRegression
import bokeh.plotting as bp
from bokeh.models import ColumnDataSource

class IdealFunctionSelector:
    def __init__(self, training_data, ideal_functions):
        """
        Initialize IdealFunctionSelector.

        Parameters:
        - training_data (pd.DataFrame): DataFrame containing training data.
        - ideal_functions (pd.DataFrame): DataFrame containing ideal functions.
        """
        self.training_data = training_data
        self.ideal_functions = ideal_functions
        self.selected_functions = None

    def calculate_deviation(self, coefficients, x_values, y_values):
        """
        Calculate the deviation between predicted and actual values.

        Parameters:
        - coefficients (np.ndarray): Coefficients of the regression model.
        - x_values (np.ndarray): X values.
        - y_values (np.ndarray): Actual Y values.

        Returns:
        - float: Sum of squared deviations.
        """
           
        if len(coefficients) != len(x_values) or len(y_values) != len(x_values):
            raise ValueError("Length mismatch between coefficients, x_values, and y_values.")
        
        predicted_values = np.polyval(coefficients[::-1], x_values)
        deviations = predicted_values - y_values
        return np.sum(deviations**2)

    def select_ideal_functions(self):
        """
        Select ideal functions based on least squares regression.
        """
        x_values = self.training_data['x'].values
        y_values = self.training_data[['y1', 'y2', 'y3', 'y4']].values.T

        selected_functions = []

        for i in range(1, min(51, len(y_values) + 1)):
            function_name = f'Y{i}'
            num_coeffs = min(i + 1, len(self.ideal_functions.columns) - 1)

            regression_model = LeastSquaresRegression(x_values, y_values[i-1])
            coefficients = regression_model.fit(degree=num_coeffs - 1)

            selected_functions.append((function_name, coefficients))

        self.selected_functions = selected_functions

    def get_selected_functions(self):
        """
        Get the selected ideal functions.

        Returns:
        - list: List of tuples containing function names and coefficients.
        """
         
        return self.selected_functions

    def predict_ideal_function_nos(self, test_data):
        """
        Predict ideal function numbers for test data.

        Parameters:
        - test_data (pd.DataFrame): DataFrame containing test data.

        Returns:
        - pd.Series: Predicted ideal function numbers.
        """
        if 'X' not in test_data.columns:
            raise ValueError("Test data must have a column named 'X'.")

        x_values = test_data['X'].values
        predicted_function_nos = []

        for x_value in x_values:
            closest_function = min(self.selected_functions, key=lambda func: abs(func[1][0] - x_value))
            predicted_function_nos.append(closest_function[0])

        return pd.Series(predicted_function_nos, name="ideal_functions_df")

    def visualize_results(self, mapped_test_data, filename="ideal_functions.html"):
        """
        Visualize test data and ideal functions using Bokeh.

        Parameters:
        - mapped_test_data (pd.DataFrame): DataFrame containing mapped test data.
        - filename (str): Name of the HTML file to save the plot.
        """
        p = bp.figure(
            title="Test Data vs Ideal Functions",
            x_axis_label="X",
            y_axis_label="Y",
        )
    
        source = ColumnDataSource(mapped_test_data)
    
        # Plot the test data
        p.circle(
            x='X',
            y='Y',
            size=10,
            color="blue",
            alpha=0.5,
            legend_label="Test Data",
            source=source,
        )

        # Plot the ideal functions
        ys_columns = list(self.ideal_functions.columns[1:])
        for col in ys_columns:
            filtered_data = mapped_test_data[mapped_test_data['ideal_functions_df'] == col]
            p.circle(
                x=filtered_data['X'],
                y=filtered_data['Y'],
                size=8,
                color="red",
                alpha=0.5,
                legend_label=col,
            )

        p.legend.location = "top_left"

        # Save the plot to an HTML file
        bp.output_file(filename)
        bp.save(p, filename)

def main():
    """
    Main function to execute the workflow.
    """
    database_url = 'sqlite:///data_fit.db'
    data_ingestion = DataIngestion(database_url)

    training_data_path = 'train.csv'
    ideal_functions_path = 'ideal.csv'
    test_data_path = 'test.csv'

    data_ingestion.preprocess_data(training_data_path, ideal_functions_path, test_data_path)

    training_data = pd.read_sql_table('train', data_ingestion.db_handler.engine)
    ideal_functions = pd.read_sql_table('ideal', data_ingestion.db_handler.engine)
    test_data = pd.read_sql_table('test', data_ingestion.db_handler.engine)

    ideal_selector = IdealFunctionSelector(training_data, ideal_functions)
    ideal_selector.select_ideal_functions()

    mapped_test_data = test_data.join(ideal_selector.predict_ideal_function_nos(test_data))
    
    filename = "ideal_functions.html"
    bp.output_file(filename)
    
    ideal_selector.visualize_results(mapped_test_data)
    
    selected_functions = ideal_selector.get_selected_functions()

    # Visualize selected functions against test data
    p_selected_functions = bp.figure(
        title="Selected Functions vs Test Data",
        x_axis_label="X",
        y_axis_label="Y",
    )

    source_selected_functions = ColumnDataSource(mapped_test_data)

    # Plot the test data
    p_selected_functions.circle(
        x='X',
        y='Y',
        size=10,
        color="blue",
        alpha=0.5,
        legend_label="Test Data",
        source=source_selected_functions,
    )

    # Plot the selected functions
    for function_name, coefficients in selected_functions:
        predicted_y_values = np.polyval(coefficients[::-1], mapped_test_data['X'])
        p_selected_functions.line(
            x=mapped_test_data['X'],
            y=predicted_y_values,
            line_width=2,
            color="red",
            alpha=0.5,
            legend_label=f'{function_name} (Selected Function)',
        )

    p_selected_functions.legend.location = "top_left"

    # Save the plot to an HTML file
    filename_selected_functions = "selected_functions_vs_test_data.html"
    bp.output_file(filename_selected_functions)
    bp.save(p_selected_functions, filename_selected_functions)


if __name__ == "__main__":
    main()