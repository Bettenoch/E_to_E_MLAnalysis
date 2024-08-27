from data_display import load_housing_data
from strat_split_data import stratified_split_data

def training_data():
    housing = load_housing_data()
    strat_train_set, strat_test_set = stratified_split_data(housing)
    # housing_test_data = strat_test_set.copy()
    housing_train_data = strat_train_set.copy()
    
    return housing_train_data
    
    # Lets look at the correlation matrix for all attributes
    
if __name__ == '__main__':
    housing_train_data = training_data()
    corr_matrix = housing_train_data.corr()
    # print(housing_train_data.info())
    print(corr_matrix["median_house_value"].sort_values(ascending=False))