import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker, declarative_base
import bokeh.plotting as bp
import bokeh.transform as bt
from bokeh.models import ColumnDataSource


Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)
    Y1 = Column(Float)
    Y2 = Column(Float)
    Y3 = Column(Float)
    Y4 = Column(Float)

class IdealFunctions(Base):
    __tablename__ = 'ideal_functions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)

    # Dynamic creation of Y columns using a loop
    for i in range(1, 51):
        locals()[f'Y{i}'] = Column(Float)
 
class LeastSquaresRegression:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def calculate_least_squares(X, y):
    model = LeastSquaresRegression()

    # Fit the model to the data
    model_fit = model.fit(X, y)

    # Predict Y values using the fitted model
    y_pred = model_fit.predict(X)

    # Calculate the mean squared error
    mse = mean_squared_error(y, y_pred)

    return model_fit, mse

class IdealFunctionMapper:
    def __init__(self, training_data, ideal_functions_df):
        self.training_data = training_data
        self.ideal_functions_df = ideal_functions_df
        self.fitted_models = {}

    def fit_models(self):
        for column in self.training_data.columns[1:]:
            x_train = self.training_data[['x']].values.reshape(-1, 1)
            y_train = self.training_data[column].values

            model = LeastSquaresRegression()
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

def main():
    # Create a database engine
    engine = create_engine("sqlite:///ideal_functions.db")

    # Create the database tables
    Base.metadata.create_all(engine)

    # Load the training and ideal function data sets
    training_data = pd.read_csv("train.csv")
    ideal_functions_df = pd.read_csv("ideal.csv")

    # Create an IdealFunctionMapper object
    mapper = IdealFunctionMapper(training_data, ideal_functions_df)

    # Fit the models
    mapper.fit_models()

    # Load the test data
    test_data = pd.read_csv("test.csv")

    # Predict the ideal function numbers for the test data
    mapped_test_data = test_data.join(mapper.predict_ideal_function_nos(test_data))

    # Visualize the results
    mapper.visualize_results(mapped_test_data)

if __name__ == '__main__':
    main()