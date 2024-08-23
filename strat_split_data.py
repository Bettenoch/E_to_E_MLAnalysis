import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split_data(housing, test_size=0.2, random_state=42):
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=random_state,)
    strat_splits = []
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n, strat_splits])
        strat_train_set, strat_test_set = strat_splits[0], strat_splits[0]
        
        return strat_train_set, strat_test_set
            
    