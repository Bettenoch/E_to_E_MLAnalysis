import os
import pandas as pd
from test_set import split_train_test


def load_housing_data(housing_path="datasets/housing"):
    """
    Loads the housing data from the specified path into a Pandas DataFrame.

    Parameters:
        housing_path (str): The directory where the housing data CSV file is located.

    Returns:
        pd.DataFrame: The loaded housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    housing_data = pd.read_csv(csv_path)
    return housing_data

# Example usage
if __name__ == "__main__":
    housing_data = load_housing_data()
    # print(housing_data.head())
    # print(housing_data.info())
    # print(housing_data["ocean_proximity"].value_counts())
    # print(housing_data.describe())
    train_set, test_set = split_train_test(housing_data, 0.2)
    print(len(train_set))
    print (len(test_set))
    
    

