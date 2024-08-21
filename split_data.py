from sklearn.model_selection import train_test_split

def split_housing_data(housing, test_size=0.2, random_state=42):
    """
    Splits the housing data into training and test sets.

    Parameters:
        housing (pd.DataFrame): The loaded housing data.
        test_size (float): The proportion of the data to include in the test set.
        random_state (int): Random seed to ensure reproducibility.

    Returns:
        tuple: A tuple containing the training set and the test set as Pandas DataFrames.
    """
    train_set, test_set = train_test_split(housing, test_size=test_size, random_state=random_state)
    
    # Check for missing values in 'total_bedrooms' in the test set
    missing_values = test_set["total_bedrooms"].isnull().sum()
    print(f"Missing values in 'total_bedrooms' in test set: {missing_values}")
    
    return train_set, test_set
