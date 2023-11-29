#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the X_train.csv file from a local location, remember using this slash / not \
X_train = pd.read_csv(r"C:/Users/84981/Dropbox/PC/Documents/Documents_old/Nha research_2018/FTC PROJECT/INTERATION FIELDS_FTC_197 PEPTIDES/df_X_train_S2.1.csv")
X_train = X_train.iloc[:,1:]
X_train.shape


# In[3]:


# Read the X_test.csv file from a local location, remember using this slash / not \
X_test = pd.read_csv(r"C:/Users/84981/Dropbox/PC/Documents/Documents_old/Nha research_2018/FTC PROJECT/INTERATION FIELDS_FTC_197 PEPTIDES/df_X_test_S2.1.csv")
X_test = X_test.iloc[:,1:]
X_test.head(7)


# In[4]:


y_train=pd.read_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_train_S2.1.csv')
y_train = y_train.iloc[:,1]
y_train


# In[5]:


y_test=pd.read_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_test_S2.1.csv')
y_test = y_test.iloc[:,1]
y_test


# # Define Regression model before hyperparatmeter tuning

# In[6]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[7]:


# Define PLS model before hyperparatmeter tuning
from sklearn.cross_decomposition import PLSRegression
model_1 = PLSRegression(n_components= 3)
pipeline_1 = Pipeline(steps=[('scaler', StandardScaler()),('model', model_1)])  
# R2, MSE and RMSE of the training set
y_pred_train = pipeline_1.fit(X_train, y_train).predict(X_train)
print("R2_train: %.3f" % r2_score(y_train,y_pred_train))
print("MSE_train: %.3f"% mean_squared_error(y_train,y_pred_train))
print("RMSE_train: %.3f"% np.sqrt(mean_squared_error(y_train,y_pred_train)))
# R2, MSE and RMSE of the test set
y_pred_test = pipeline_1.fit(X_train, y_train).predict(X_test)
print("R2_test: %.3f" % r2_score(y_test,y_pred_test))
print("MSE_test: %.3f" % mean_squared_error(y_test,y_pred_test))
print("RMSE_test: %.3f" % np.sqrt(mean_squared_error(y_test,y_pred_test)))


# In[8]:


pipeline_1.get_params()


# In[9]:


## Cross validation using 5 fold cross validation before hyperparatmeter tuning
from numpy import mean
from numpy import std
from numpy import absolute

# Computing 5 fold-cross validation scores
scores1 = cross_val_score(pipeline_1, X_train, y_train, scoring ='r2', cv=5, n_jobs=-1, error_score='raise')
scores2 = cross_val_score(pipeline_1, X_train, y_train, scoring ='neg_mean_squared_error', cv=5, n_jobs=-1, error_score='raise')

# force scores to be positive
scores2 = absolute(scores2)
# Computing mean of MSE and std of the GBR_cross validation scores
print('Mean R2: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Mean MSE: %.3f (%.3f)' % (mean(scores2), std(scores2)))


# # Feature selection using GridSearchCV

# In[10]:


from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

pipeline_2= Pipeline(steps=[("scaler", StandardScaler()), ('estimator', PLSRegression())])
param_grid = {'estimator__n_components': range(2, 16)}

feat_search = GridSearchCV(estimator=pipeline_2,param_grid=param_grid, scoring="neg_mean_squared_error",n_jobs= -1, cv=5)
feat_search_cv = feat_search.fit(X_train, y_train)
feat_search_df = pd.DataFrame(feat_search_cv.cv_results_)
feat_search_df


# In[11]:


feat_search_cv.best_score_


# In[12]:


feat_search_cv.best_params_


# In[13]:


n_components = feat_search_cv.best_params_.get('estimator__n_components')
n_components


# # PLSRegression model with the optimal n_components

# In[14]:


# Define PLS regression model with hyperparatmeter tuning

model_3 = PLSRegression(n_components= n_components)
pipeline_3 = Pipeline(steps=[('scaler', StandardScaler()),('model', model_3)])  
# R2, MSE and RMSE of the training set
y_pred_train = pipeline_3.fit(X_train, y_train).predict(X_train)
print("R2_train: %.3f" % r2_score(y_train,y_pred_train))
print("MSE_train: %.3f"% mean_squared_error(y_train,y_pred_train))
print("RMSE_train: %.3f"% np.sqrt(mean_squared_error(y_train,y_pred_train)))
# R2, MSE and RMSE of the test set
y_pred_test = pipeline_3.fit(X_train, y_train).predict(X_test)
print("R2_test: %.3f" % r2_score(y_test,y_pred_test))
print("MSE_test: %.3f" % mean_squared_error(y_test,y_pred_test))
print("RMSE_test: %.3f" % np.sqrt(mean_squared_error(y_test,y_pred_test)))


# In[15]:


## Cross validation using 5 fold cross validation after GridSearchCV hyperparatmeter tuning
from numpy import mean
from numpy import std
from numpy import absolute

# Computing 5 fold-cross validation scores
scores3 = cross_val_score(pipeline_3, X_train, y_train, scoring ='r2', cv=5, n_jobs=-1, error_score='raise')
scores4 = cross_val_score(pipeline_3, X_train, y_train, scoring ='neg_mean_squared_error', cv=5, n_jobs=-1, error_score='raise')

# force scores to be positive
scores4 = absolute(scores4)
# Computing mean of r2 and MSE and std of the GBR_cross validation scores
print('Mean R2: %.3f (%.3f)' % (mean(scores3), std(scores3)))
print('Mean MSE: %.3f (%.3f)' % (mean(scores4), std(scores4)))

