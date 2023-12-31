from sqlalchemy import create_engine, Column, Float, Integer
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

class DatabaseHandler:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.create_tables()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def load_data(self, file_path, table):
        csv_loader = CSVLoader(file_path)
        csv_loader.load_data()
        data = csv_loader.get_data()

        if data is not None:
            data.to_sql(table.__tablename__, self.engine, if_exists='replace', index=False)
        else:
            print(f"No data loaded from {file_path}")

def main():
    database_url = 'sqlite:///data_fit.db'
    db_handler = DatabaseHandler(database_url)

    # Load training data into the database
    db_handler.load_data('train.csv', TrainingData)

    # Load ideal functions data into the database
    db_handler.load_data('ideal.csv', IdealFunctions)

    # Load test data into the database
    db_handler.load_data('test.csv', TestData)

if __name__ == "__main__":
    main()
