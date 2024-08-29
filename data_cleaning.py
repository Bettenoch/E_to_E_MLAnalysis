import pandas as pd
from training_data import training_data
from sklearn.impute import SimpleImputer
import numpy as np

def data_cleaning():
    # Load the housing data
    housing_data = training_data()
    
    # Separate the target label
    housing = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    # Separate the numeric and categorical data
    housing_num = housing.select_dtypes(include=[np.number])
    housing_cat = housing.select_dtypes(exclude=[np.number])

    # Impute missing values for the numeric data
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    # Reintegrate the categorical data back to the cleaned numeric data
    housing_tr = pd.concat([housing_tr, housing_cat], axis=1)

    return housing_tr, housing_labels

if __name__ == '__main__':
    housing_tr, housing_labels = data_cleaning()
    # print(housing_tr.head())
    
    housing_cat = housing_tr[["ocean_proximity"]]
    print(housing_cat.head())
 
