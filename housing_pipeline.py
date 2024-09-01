import pandas as pd
from data_cleaning import data_cleaning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

def housing_pipeline():
    # Get the cleaned data
    housing_tr, housing_labels = data_cleaning()

    # List of numeric and categorical attributes
    num_attribs = housing_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # Pipeline for numerical attributes
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # Full pipeline: numeric and categorical
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse_output=False), cat_attribs),
    ])

    # Prepare the housing data
    housing_prepared = full_pipeline.fit_transform(housing_tr)

    return housing_prepared, housing_labels

if __name__ == '__main__':
    housing_prepared, housing_labels = housing_pipeline()
    print(housing_prepared.shape)  # Optional: to check the shape of the prepared data
