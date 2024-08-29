from data_display import load_housing_data
from strat_split_data import stratified_split_data
from pandas.plotting import scatter_matrix
def training_data():
    housing = load_housing_data()
    strat_train_set, strat_test_set = stratified_split_data(housing)
    # housing_test_data = strat_test_set.copy()
    housing_train_data = strat_train_set.copy()
    # Select only the numeric columns for correlation matrix calculation
    # housing_train_data_numeric = housing_train_data.select_dtypes(include=[float, int])
    
    return housing_train_data
    
    # Lets look at the correlation matrix for all attributes
    
    
if __name__ == '__main__':
    housing_data = training_data()
    corr_matrix = housing_data.corr(numeric_only=True)
    # The correlation coefficient ranges from -1 to 1. Closer to 1 means strong positive correlation and closer to -1 mean negative correlation
    # median_house_value    1.000000
    # median_income         0.687151
    # total_rooms           0.135140
    # housing_median_age    0.114146
    # households            0.064590
    # total_bedrooms        0.047781
    # population           -0.026882
    # longitude            -0.047466
    # latitude             -0.142673
    # Name: median_house_value, dtype: float64
    
    result = corr_matrix["median_house_value"].sort_values(ascending=False)
    # print( result)
    
    attributes = ["median_house_value", "total_rooms", "median_income", "housing_median_age"]
    scatter_matrix(housing_data[attributes], figsize=(12, 8))
    
    housing_data["rooms_per_household"] = housing_data["total_rooms"] / housing_data["households"]  
    housing_data["bedrooms_per_room"] = housing_data["total_bedrooms"] /housing_data["total_rooms"]
    housing_data["population_per_household"] =housing_data["population"] / housing_data["households"]
    
    corr_matrix2 = housing_data.corr(numeric_only=True)
    result2 = corr_matrix["median_house_value"].sort_values(ascending=False) 
    print(result2)