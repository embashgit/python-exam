import pandas as pd
import bokeh.plotting as bp
import bokeh.transform as bt
from bokeh.models import ColumnDataSource
from least_square_regression import LeastSquaresRegressionAlt

class IdealFunctionMapper:
    def __init__(self, training_data, ideal_functions_df):
        self.training_data = training_data
        self.ideal_functions_df = ideal_functions_df
        self.fitted_models = {}

    def fit_models(self):
        for column in self.training_data.columns[1:]:
            x_train = self.training_data[['x']].values.reshape(-1, 1)
            y_train = self.training_data[column].values

            model = LeastSquaresRegressionAlt()
            model.fit(x_train, y_train)

            self.fitted_models[column] = model

    def predict_ideal_function_nos(self, test_data):
        ideal_function_nos = []

        for column in self.training_data.columns[1:]:
            model = self.fitted_models[column]
            predicted_ideal_function_no = model.predict(test_data[['x']])

            ideal_function_nos.append(predicted_ideal_function_no)

        return pd.Series(ideal_function_nos, name="ideal_function_no")
    
    def visualize_results(self, mapped_test_data, filename="ideal_functions.html"):
        p = bp.figure(
            title="Test Data vs Ideal Functions",
            x_axis_label="x",
            y_axis_label="y",
        )
    
        # Create a ColumnDataSource
        source = ColumnDataSource(mapped_test_data)
    
        # Plot the test data
        p.circle(
            x='x',
            y='y',
            size=10,
            color="blue",
            alpha=0.5,
            legend_label="Test Data",
            source=source,
        )
    
        # Plot the ideal functions
        p.multi_line(
            x='x',
            y=list(self.ideal_functions_df.columns[1:]),
            source=source,
            line_width=2,
            color="red",
            alpha=0.5,
            legend_label=list(self.ideal_functions_df.columns[1:]),
        )
    
        p.legend.location = "top_left"
    
        # Save the plot to an HTML file
        bp.save(p, filename)