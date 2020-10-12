## The inputs that are required to run the script can be obtained by  typing "python shapanalysis.py -h"
#An example to run the script is :
#python shapanalysis.py -i all_data_combined_MaskEdit_2.csv -o output -t 0.2 -f 20 -l 0.08 -n 525 -d 7 -w 5 -g 0 -s 0.85 -c 0.75 -a 1

import pandas as pd
import xgboost as xgb
import shap
import pickle

from sklearn import model_selection
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sys import argv
import argparse
from matplotlib import cm
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument('-n','--nest', dest="n_est_opt", help='n_estimators',type=int,required=True)
parser.add_argument('-l','--lr', dest = "lr",help='learning_rate',type=float, required=True)
parser.add_argument('-d','--mdepth', dest = "maxdepth_opt",help='max_depth',type=int,required=True)
parser.add_argument('-w','--mcweight', dest = "min_child_weight_opt", help='min_child_weight',type=float,required=True)
parser.add_argument('-g','--gamma', dest='gamma_opt', help = "gamma", type=float,required=True)
parser.add_argument('-s','--subsample', dest="subsample_opt", type=float,required=True)
parser.add_argument('-c','--cols_bytree', dest= "colsample_bytree_opt", help="colsample_bytree",type=float,required=True)
parser.add_argument('-a','--alpha', dest= "reg_alpha_opt", help="reg_alpha",type=float,required=True)
parser.add_argument('-f','--colnum', dest= "colnum", help="target column number",type=int,required=True)
parser.add_argument('-o','--output', dest= "out", help="output file name",type=str,required=True)
parser.add_argument('-i','--input', dest= "infile", help="input file name",type=str,required=True)
parser.add_argument('-t','--testsize', dest= "testsize", help="test set size as fraction",type=float,required=True)

options = parser.parse_args()

string = "# python"
for ele in argv:
        string +=  " "+ele
print ("\n",string,"\n")
# load JS visualization code to notebook
shap.initjs()

X = pd.read_csv(options.infile, header='infer',delimiter=",",usecols=([0,1]+[i for i in range (2,12)]),index_col=[0,1])
Y = pd.read_csv(options.infile, header='infer',delimiter=",",usecols=[0,1,options.colnum],index_col=[0,1])
X.to_csv('inputs.csv')

print ("Features and target columns")
print (X,Y)
print ("Features",X.columns)
no_of_inputs = X.shape[1]
print ("Number of features",no_of_inputs)



test_size = options.testsize
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=123)


####  model

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = options.lr, seed=27,n_estimators=options.n_est_opt,max_depth=options.maxdepth_opt,min_child_weight=options.min_child_weight_opt,gamma=options.gamma_opt,subsample=options.subsample_opt, colsample_bytree=options.colsample_bytree_opt,reg_alpha=options.reg_alpha_opt) 


xg_reg.fit(X_train,y_train)
print ("XGB Parameters",xg_reg.get_xgb_params())
y_pred = xg_reg.predict(X_test)
y_all_pred = xg_reg.predict(X)
y_train_pred = xg_reg.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE: %f" % (rmse))
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Train RMSE: %f" % (rmse))

plt.scatter(y_train,y_train_pred,label="training set",alpha=0.5)
plt.scatter(y_test,y_pred,label="test set",alpha=0.5)
plt.xlabel("Observed Rate",fontsize=22)
plt.ylabel("Predicted Rate",fontsize=22)
plt.legend()
plt.savefig(options.out+"pred.svg")
plt.savefig(options.out+"pred.pdf")
#plt.show()
plt.close()

print ("R2_train =",r2_score(y_train, y_train_pred),"R2_test =",r2_score(y_test, y_pred),"R2_all =",r2_score(Y,y_all_pred))

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X)

# summarize the effects of all the features
shap.summary_plot(shap_values, X,cmap=plt.get_cmap('viridis'),show=False,alpha=0.5,plot_size=(8,6))
plt.savefig(options.out+"_summaryplot.pdf",bbox_inches="tight")
plt.savefig(options.out+"_summaryplot.eps",bbox_inches="tight")
plt.savefig(options.out+"_summaryplot.svg",bbox_inches="tight")

np.savetxt(options.out+"_SHAP.csv", shap_values, delimiter=",")
np.savetxt(options.out+"_allpredFIT.csv", (np.column_stack((Y.to_numpy(),y_all_pred.reshape((y_all_pred.shape[0],1))))), delimiter=",")
np.savetxt(options.out+"_trainpredFIT.csv", (np.column_stack((y_train.to_numpy(),y_train_pred.reshape((y_train_pred.shape[0],1))))), delimiter=",")
np.savetxt(options.out+"_testpredFIT.csv", (np.column_stack((y_test.to_numpy(),y_pred.reshape((y_pred.shape[0],1))))), delimiter=",")

### k-fold cross validation
print ("K-fold cross-validation")
nsplits = int(1/test_size)
#print ("nsplits = ",nsplits)
kf = KFold(n_splits=nsplits,random_state=123,shuffle=True)
#print (kf.get_n_splits(X))
print(kf)
shap_values_cv = np.zeros((shap_values.shape[0],shap_values.shape[1]))
rmsevals = []
k = 1
all_predictions = np.column_stack((Y.to_numpy(),y_all_pred.reshape((y_all_pred.shape[0],1))))
#print (all_predictions.shape)
for train_index, test_index in kf.split(X):
#	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
	y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
	xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = options.lr, seed=27,n_estimators=options.n_est_opt,max_depth=options.maxdepth_opt,min_child_weight=options.min_child_weight_opt,gamma=options.gamma_opt,subsample=options.subsample_opt, colsample_bytree=options.colsample_bytree_opt,reg_alpha=options.reg_alpha_opt)
	xg_reg.fit(X_train,y_train)
	y_pred = xg_reg.predict(X_test)
	y_train_pred = xg_reg.predict(X_train)
	rmsetest = np.sqrt(mean_squared_error(y_test, y_pred))
	print ("Fold",k)
	print("Test RMSE: %f" % (rmsetest))
	rmsetrain = np.sqrt(mean_squared_error(y_train, y_train_pred))
	print("Train RMSE: %f" % (rmsetrain))
	explainer = shap.TreeExplainer(xg_reg)
	shap_values_new = explainer.shap_values(X)
	shap_values_cv += shap_values_new
	np.savetxt(options.out+"_"+str(k)+"_SHAP.csv", shap_values_new, delimiter=",")
	y_all_pred = xg_reg.predict(X)
	all_predictions = np.column_stack((all_predictions,y_all_pred.reshape((y_all_pred.shape[0],1))))
	#print (all_predictions.shape)
	rmsevals.append([rmsetrain,rmsetest])
	k+=1
#	plt.scatter(Y,xg_reg.predict(X))
#	plt.show()
shap_values_cv = shap_values_cv/nsplits
header = ' '.join(list(X.columns))
np.savetxt(options.out+"_avg_SHAP.csv", shap_values_cv,header=header)
all_predictions = np.column_stack((all_predictions,np.mean(all_predictions[:,2:2+nsplits],axis=1)))
np.savetxt(options.out+"_allpred_cv.csv", all_predictions,header="observed_rate predicted_rate CV1_pred CV2_pred CV3_pred CV4_pred CV5_pred CV_avg_pred")
rmsevals = np.array(rmsevals)
print ("Mean train rmse=",np.mean(rmsevals[:,0]),"Std train rmse=",np.std(rmsevals[:,0]))
print ("Mean test rmse=",np.mean(rmsevals[:,1]),"Std test rmse=",np.std(rmsevals[:,1]))
