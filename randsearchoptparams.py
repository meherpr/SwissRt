##Inputs to run the python script
#argv[1] : Filename
#argv[2] : Column number for output column (rate for next week)
#argv[3] : random state
#argv[4] : Filename to save output of optimization
#Example : python3.7 randsearchoptparams.py all_data_combined_MaskEdit_2.csv 20 100 randsearchout_100.dat

import time
import numpy as np
import xgboost  as  xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sys import argv
import pandas as pd
from sklearn import model_selection

start_time=time.time()

print ("python3.7 randsearchoptparams.py",argv[1],argv[2],argv[3],argv[4])
#### Load data
X = pd.read_csv(argv[1], header='infer',delimiter=",",usecols=([0,1]+[i for i in range (2,12)]),index_col=[0,1])
Y = pd.read_csv(argv[1], header='infer',delimiter=",",usecols=[0,1,int(argv[2])],index_col=[0,1])

print (X.columns)
print (X,Y,X.index)


#### Create X and Y training data
test_size = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=123)

# grid search
model = xgb.XGBRegressor(objective ='reg:squarederror')

param_grid = {
        'max_depth': [3,5,7,9,15,20,25,50],
        'min_child_weight': [1,3,5,7,9,15,20,25,50],
        'gamma': np.arange(0.0,20.0,0.05),
        'learning_rate': np.arange(0.01,0.25,0.01),
        'subsample': np.arange(0.1,1.0,0.05),
        'colsample_bytree': np.arange(0.1,1.0,0.05),
	'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 73)],
	'reg_alpha' :[0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100] 
}
kfold = KFold(n_splits=5, shuffle=True, random_state=10)
grid_search = RandomizedSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_iter = 2000, cv=kfold,return_train_score=True,random_state=int(argv[3]))
grid_result = grid_search.fit(X_train,y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
params = grid_result.cv_results_[ 'params' ]
print ((pd.DataFrame(params)).to_string())
print ((pd.DataFrame(grid_result.cv_results_)).to_string())
cv_df = pd.DataFrame(grid_result.cv_results_)
cv_df = cv_df.loc[:, cv_df.columns != 'params']
cv_df.to_csv(argv[4])
print(time.time()-start_time)
