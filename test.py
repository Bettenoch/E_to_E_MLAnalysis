import pandas as pd
from training_data import training_data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def data_cleaning():
    # Load the housing data
    housing_data = training_data()
    
    # Separate the target label
    housing = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    # Separate the numeric and categorical data
    housing_num = housing.select_dtypes(include=[np.number])
    housing_cat = housing[["ocean_proximity"]]  # Extract only the ocean_proximity column

    # Impute missing values for the numeric data
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    # Apply OneHotEncoder to the categorical data
    cat_encoder = OneHotEncoder(sparse_output=False)  # Set sparse_output=False to get a dense array
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    # Convert the OneHotEncoder output to a DataFrame
    housing_cat_1hot_df = pd.DataFrame(housing_cat_1hot, 
                                       columns=cat_encoder.get_feature_names_out(),
                                       index=housing.index)
    
    # Reintegrate the transformed categorical data back to the cleaned numeric data
    housing_tr = pd.concat([housing_tr, housing_cat_1hot_df], axis=1)

    return housing_tr, housing_labels

if __name__ == '__main__':
    housing_tr, housing_labels = data_cleaning()
    print(housing_tr.head())  # Optional: to check the first few rows of the cleaned data
