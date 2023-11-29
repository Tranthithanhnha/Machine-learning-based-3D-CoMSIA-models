#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[56]:


# Read the X_train.csv file from a local location, remember using this slash / not \
X_train = pd.read_csv(r"C:/Users/84981/Dropbox/PC/Documents/Documents_old/Nha research_2018/FTC PROJECT/INTERATION FIELDS_FTC_197 PEPTIDES/df_X_train_S2.1.csv")
X_train = X_train.iloc[:,1:]
X_train.shape


# In[57]:


# Read the X_test.csv file from a local location, remember using this slash / not \
X_test = pd.read_csv(r"C:/Users/84981/Dropbox/PC/Documents/Documents_old/Nha research_2018/FTC PROJECT/INTERATION FIELDS_FTC_197 PEPTIDES/df_X_test_S2.1.csv")
X_test = X_test.iloc[:,1:]
X_test.head(7)


# In[58]:


y_train=pd.read_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_train_S2.1.csv')
y_train = y_train.iloc[:,1]
y_train


# In[59]:


y_test=pd.read_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_test_S2.1.csv')
y_test = y_test.iloc[:,1]
y_test


# # Define Regression model before hyperparatmeter tuning

# In[60]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[61]:


# Define Gradient Boosting regression model before hyperparatmeter tuning
from sklearn.ensemble import GradientBoostingRegressor
model_1 = GradientBoostingRegressor(random_state = 1)
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


# In[62]:


pipeline_1.get_params()


# In[63]:


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


# # Hyperparameter tuning using GridSearchCV

# In[64]:


## Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
model_2 = GradientBoostingRegressor(random_state = 1)
pipeline_2 = Pipeline(steps=[('scaler', StandardScaler()),('model', model_2)])
param_grid = {'model__max_depth' : [2, 3, 5],
              'model__learning_rate': [0.01, 0.1, 0.5], 
              'model__n_estimators': [100, 500, 1000],                            
              'model__subsample': [0.1, 0.5, 1]}
grid_search = GridSearchCV(pipeline_2, param_grid=param_grid,
                            scoring='neg_mean_squared_error',n_jobs= -1)
grid_search_cv = grid_search.fit(X_train, y_train)
GridSearch_df = pd.DataFrame(grid_search_cv.cv_results_)


# In[65]:


grid_search_cv.best_score_


# In[66]:


grid_search_cv.best_params_


# In[67]:


d = grid_search_cv.best_params_
d1 = {'model__learning_rate': 'learning_rate',
 'model__max_depth': 'max_depth',
 'model__n_estimators': 'n_estimators',
 'model__subsample': 'subsample'}
d2= dict((d1[key], value) for (key, value) in d.items())   
d2


# In[68]:


# Define Gradient Boosting regression model using optimized hyperparameters from GridSearchCV
# the grid_search_cv.best_params_ have to be unpacked using the sign **
model_3 = GradientBoostingRegressor(**d2, random_state = 1)
pipeline_3 = Pipeline(steps=[('scaler', StandardScaler()), ('model', model_3)])

# R2, MSE and RMSE of the training set
y_pred_train2 = pipeline_3.fit(X_train, y_train).predict(X_train)
print("R2_train2: %.3f" % r2_score(y_train,y_pred_train2))
print("MSE_train2: %.3f"% mean_squared_error(y_train,y_pred_train2))
print("RMSE_train2: %.3f"% np.sqrt(mean_squared_error(y_train,y_pred_train2)))
# R2, MSE and RMSE of the test set
y_pred_test2 = pipeline_3.fit(X_train, y_train).predict(X_test)
print("R2_test2: %.3f" % r2_score(y_test,y_pred_test2))
print("MSE_test2: %.3f" % mean_squared_error(y_test,y_pred_test2))
print("RMSE_test2: %.3f"% np.sqrt(mean_squared_error(y_test,y_pred_test2)))


# In[69]:


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


# ## Hyperparameter tuning using RandomizedSearchCV

# In[70]:


## Hyperparameter tuning using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, truncnorm, randint
from sklearn.ensemble import GradientBoostingRegressor
model_4 = GradientBoostingRegressor(random_state = 1)
pipeline_4 = Pipeline(steps=[('scaler', StandardScaler()),('model', model_4)])
param_distributions = {'model__max_depth'    : randint(2, 10),
                       'model__learning_rate': uniform(),                      
                        'model__n_estimators' : randint(100, 1000),                       
                        'model__subsample'    : uniform()}
RandomSearch = RandomizedSearchCV(pipeline_4, param_distributions=param_distributions,
                                scoring='neg_mean_squared_error', n_iter=20, random_state=1, n_jobs= -1)
random_search_cv = RandomSearch.fit(X_train, y_train)
RandomSearch_df = pd.DataFrame(random_search_cv.cv_results_)
RandomSearch_df


# In[71]:


random_search_cv.best_params_


# In[72]:


random_search_cv.best_score_


# In[73]:


dic = random_search_cv.best_params_
dic1 = {'model__learning_rate': 'learning_rate',
 'model__max_depth': 'max_depth',
 'model__n_estimators': 'n_estimators',
 'model__subsample': 'subsample'}
dic2= dict((dic1[key], value) for (key, value) in dic.items())   
dic2


# In[74]:


# Define Gradient Boosting regression model using optimized hyperparameters from RandomizedSearchCV
model_5 = GradientBoostingRegressor(**dic2, min_samples_split=10, min_samples_leaf=10, random_state = 1)
pipeline_5 = Pipeline(steps=[('scaler', StandardScaler()), ('model', model_5)])
# R2, MSE and RMSE of the training set
y_pred_train3 = pipeline_5.fit(X_train, y_train).predict(X_train)
print("R2_train3: %.3f" % r2_score(y_train,y_pred_train3))
print("MSE_train3: %.3f" % mean_squared_error(y_train,y_pred_train3))
print("RMSE_train3: %.3f" % np.sqrt(mean_squared_error(y_train,y_pred_train3)))
# R2, MSE and RMSE of the test set
y_pred_test3 = pipeline_5.fit(X_train, y_train).predict(X_test)
print("R2_test3: %.3f" % r2_score(y_test,y_pred_test3))
print("MSE_test3: %.3f" % mean_squared_error(y_test,y_pred_test3))
print("RMSE_test3: %.3f"% np.sqrt(mean_squared_error(y_test,y_pred_test3)))


# In[75]:


## Cross validation using 5 fold cross validation after RandomizedSearchCV hyperparatmeter tuning 
from numpy import mean
from numpy import std
from numpy import absolute

# Computing 5 fold-cross validation scores
scores5 = cross_val_score(pipeline_5, X_train, y_train, scoring ='r2', cv=5, n_jobs=-1, error_score='raise')
scores6 = cross_val_score(pipeline_5, X_train, y_train, scoring ='neg_mean_squared_error', cv=5, n_jobs=-1, error_score='raise')

# force scores to be positive
scores6 = absolute(scores6)
# Computing mean of r2 and MSE and std of the GBR_cross validation scores
print('Mean R2: %.3f (%.3f)' % (mean(scores5), std(scores5)))
print('Mean MSE: %.3f (%.3f)' % (mean(scores6), std(scores6)))


# In[ ]:




