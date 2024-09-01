import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from data_cleaning import data_cleaning

# Custom transformer for clustering similarity
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1., random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        self.distances = self.kmeans.transform(X)
        return self

    def transform(self, X, y=None):
        distances = self.kmeans.transform(X)
        return np.exp(-self.gamma * distances)
    
    def get_feature_names_out(self, input_features=None):
        return [f"geo__Cluster {i} similarity" for i in range(self.n_clusters)]

# Custom ratio transformer
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

# Create the transformation pipeline
def transformation_pipeline():
    housing_tr, housing_labels = data_cleaning()
    
    # Define pipelines
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", make_pipeline(SimpleImputer(strategy="median"),
                              FunctionTransformer(np.log, feature_names_out="one-to-one"),
                              StandardScaler()), 
                ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", ClusterSimilarity(n_clusters=10, gamma=1., random_state=42), ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=num_pipeline)  # Apply the num_pipeline to the remainder of the columns

    # Transform the data
    housing_prepared = preprocessing.fit_transform(housing_tr)
    
    # Convert the transformed data to a DataFrame (optional)
    housing_prepared_df = pd.DataFrame(
        housing_prepared, 
        columns=preprocessing.get_feature_names_out(),
        index=housing_tr.index)
    
    return housing_prepared_df, housing_labels


if __name__ == '__main__':
    housing_prepared_df, housing_labels = transformation_pipeline()
    print(housing_prepared_df.shape)
    print(housing_prepared_df.head())  # Optional: to check the first few rows of the transformed data
