from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd

Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'train'
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)
    Y1 = Column(Float)
    Y2 = Column(Float)
    Y3 = Column(Float)
    Y4 = Column(Float)

class TestData(Base):
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)
    Y = Column(Float)

class IdealFunctions(Base):
    __tablename__ = 'ideal'
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)

    # Dynamic creation of Y columns using a loop
    for i in range(1, 51): 
        locals()[f'Y{i}'] = Column(Float)

class DatabaseHandler:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.create_tables()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def load_data(self, file_path, table):
        data = pd.read_csv(file_path)
        data.to_sql(table.__tablename__, self.engine, if_exists='replace', index=False)

class CSVLoader:
    def __init__(self, filename):
        self.filename = filename
        self.data = None

    def load_data(self):
        try:
            # Load data from CSV into a Pandas DataFrame
            self.data = pd.read_csv(self.filename)
            print(f"Data loaded successfully from {self.filename}")

        except Exception as e:
            print(f"Error loading data from {self.filename}: {e}")

    def get_data(self):
        return self.data

class DataIngestion:
    def __init__(self, database_url):
        self.database_url = database_url
        self.db_handler = DatabaseHandler(database_url)

    def preprocess_data(self, training_data_path, ideal_functions_path, test_data_path):
        # Load training data into the database
        self.db_handler.load_data(training_data_path, TrainingData)

        # Load ideal functions data into the database
        self.db_handler.load_data(ideal_functions_path, IdealFunctions)

        # Load test data line by line
        test_data_loader = CSVLoader(test_data_path)
        test_data_loader.load_data()
        test_data = test_data_loader.get_data()

        if test_data is not None:
            self.load_test_data(test_data)
        else:
            print(f"No test data loaded from {test_data_path}")

    def load_test_data(self, test_data):
        Session = sessionmaker(bind=self.db_handler.engine)
        session = Session()

        try:
            for index, row in test_data.iterrows():
                test_instance = TestData(X=row['x'], Y=row['y'])
                session.add(test_instance)
                session.commit()

            print("Test data loaded successfully.")

        except Exception as e:
            session.rollback()
            print(f"Error loading test data: {e}")

        finally:
            session.close()

