#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the csv file from a local location, remember using this slash / not \
df = pd.read_csv("C:/Users/84981/Dropbox/PC/Documents/Documents_old/Nha research_2018/FTC PROJECT/INTERATION FIELDS_FTC_197 PEPTIDES/FTC_All Interaction Fields.csv")
df.head(7)


# In[3]:


# Count no. of columns and rows in the table
df.shape


# In[4]:


#Count the empty (NaN, NAN, na) values in each column
df.isna().sum()


# In[5]:


# Drops any column with missing values
df = df.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
# Drops any row with missing values
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df.shape


# In[6]:


# Remove columns with all 0s values from the data frame
df = df.loc[:, (df != 0).any(axis=0)]
df.shape


# In[7]:


# # Remove colums containing less than 5 values
# set the number of unique values below which columns will be removed
unique_threshold = 5

# identify columns with a number of unique values less than or equal to the threshold
cols_to_remove = [col for col in df.columns if df[col].nunique() <= unique_threshold]

# remove the identified columns from X_train
df.drop(cols_to_remove, axis=1, inplace=True)
df.shape


# In[8]:


df.describe()


# In[9]:


df['Activity'].min()


# In[10]:


df['Activity'].max()


# In[11]:


#Look at the data types 
df.dtypes


# In[12]:


df_X = df.iloc[:, 2:]
df_X


# In[13]:


df_y = df.iloc[:, 1]
df_y


# In[14]:


#Create correlation matrix
corr_matrix= df_X.corr().abs()
corr_matrix


# In[15]:


# Note that Correlation matrix will be mirror image about the diagonal and all the diagonal elements will be 1. 
# Select upper triangle of correlation matrix
upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
upper_corr


# In[16]:


# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper_corr.columns if any(upper_corr[column] > 0.95)]
# Drop features with correlation greater than 0.95
df_drop_corr = df_X.drop(df[to_drop], axis=1)
df_drop_corr


# In[17]:


X = df_drop_corr
X


# In[18]:


y = df.iloc[:, 1]
y


# In[19]:


# Split dataset into traning and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[20]:


df_X_train= pd.DataFrame(X_train)
df_X_train


# In[21]:


df_X_test= pd.DataFrame(X_test)
df_X_test


# # Import sklearn regressors and modules 

# In[22]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,HistGradientBoostingRegressor 


# # Feature selection using RFE and GridSearchCV

# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostRegressor
pipe = Pipeline(steps=[("scaler", StandardScaler()), ('feat_sel', RFE(estimator= AdaBoostRegressor (random_state=1), step=20)),
                       ('model', AdaBoostRegressor(random_state=1))])
param_grid = {
                       # specify here the number of features you want to test using the RFE method
                       'feat_sel__n_features_to_select': [5, 7, 9,  11,  13,  15], 
                       # ... other parameters you want to search ...
                      }

feat_search = GridSearchCV(estimator=pipe,
                            param_grid=param_grid, scoring="neg_mean_squared_error",n_jobs= -1)
feat_search_cv = feat_search.fit(X_train, y_train)
feat_search_df = pd.DataFrame(feat_search_cv.cv_results_)
feat_search_df


# In[24]:



feat_search_cv.best_params_


# In[25]:


feat_search_cv.best_score_


# In[26]:


n_features_to_select = feat_search_cv.best_params_.get('feat_sel__n_features_to_select')
n_features_to_select 


# In[27]:


rfe = RFE(AdaBoostRegressor(), n_features_to_select= n_features_to_select, step=20 )
selector = rfe.fit(X_train, y_train)


# In[28]:


selected_features = selector.get_feature_names_out(input_features=None)
selected_features


# In[29]:


X_train =selector.transform(X_train)
X_test = selector.transform(X_test)


# In[30]:


df_X_train_selected = pd.DataFrame(X_train)
df_X_train_selected 


# In[31]:


df_X_train.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\df_X_train_S5.3.csv')


# In[32]:


y_train.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_train_S5.3.csv')


# In[33]:


df_X_test_selected = pd.DataFrame(X_test)
df_X_test_selected 


# In[34]:


df_X_test.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\df_X_test_S5.3.csv')


# In[35]:


y_test.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_test_S5.3.csv')


# ## Define Regression model before hyperparatmeter tuning

# In[36]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[37]:


# Define AdaBoost regression model before hyperparatmeter tuning
from sklearn.ensemble import AdaBoostRegressor
model_1 = AdaBoostRegressor(n_estimators=100, random_state=1)
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


# In[38]:


pipeline_1.get_params()


# In[39]:


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


# ## Hyperparameter tuning using GridSearchCV

# In[40]:


## Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
model_2 = AdaBoostRegressor(random_state = 1)
pipeline_2 = Pipeline(steps=[('scaler', StandardScaler()),('model', model_2)])

param_grid = {'model__n_estimators': [50, 100,200, 500, 800, 1000],    
              'model__learning_rate': [0.01, 0.05, 0.1, 0.5, 1]}


grid_search = GridSearchCV(pipeline_2, param_grid=param_grid,
                            scoring='neg_mean_squared_error',n_jobs= -1)
grid_search_cv = grid_search.fit(X_train, y_train)
GridSearch_df = pd.DataFrame(grid_search_cv.cv_results_)
GridSearch_df


# In[43]:


GridSearch_df.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\GridSearch_df_S5.3.csv')


# In[44]:


-grid_search_cv.best_score_


# In[45]:


grid_search_cv.best_params_


# In[46]:


d = grid_search_cv.best_params_
d1 = {'model__learning_rate': 'learning_rate',
      'model__n_estimators': 'n_estimators' }
d2= dict((d1[key], value) for (key, value) in d.items())   
d2


# In[47]:


# Define AdaBoost regression model using optimized hyperparameters from GridSearchCV
# the grid_search_cv.best_params_ have to be unpacked using the sign **
model_3 = AdaBoostRegressor(**d2, random_state = 1)
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


# In[48]:


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


# # Hyperparameter tuning with RandomizedSerearchCV

# In[49]:


## Hyperparameter tuning using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, truncnorm, randint
from sklearn.ensemble import AdaBoostRegressor
model_4 = AdaBoostRegressor(random_state = 1)
pipeline_4 = Pipeline(steps=[('scaler', StandardScaler()),('model', model_4)])

param_distributions = {'model__n_estimators' : randint(100, 1000),
                       'model__learning_rate':  uniform()}

RandomSearch = RandomizedSearchCV(pipeline_4, param_distributions=param_distributions,
                                scoring='neg_mean_squared_error', n_iter=20, random_state=1, n_jobs= -1)
random_search_cv = RandomSearch.fit(X_train, y_train)
RandomSearch_df = pd.DataFrame(random_search_cv.cv_results_)
RandomSearch_df


# In[50]:


RandomSearch_df.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\RandomSearch_df_S5.3.csv')


# In[51]:


random_search_cv.best_params_


# In[52]:


random_search_cv.best_score_


# In[53]:


dic = random_search_cv.best_params_
dic1 = {'model__learning_rate': 'learning_rate',
      'model__n_estimators': 'n_estimators' }
dic2= dict((dic1[key], value) for (key, value) in dic.items())   
dic2


# In[54]:


# Define AdaBoost regression model using optimized hyperparameters from RandomizedSearchCV
model_5 = AdaBoostRegressor(**dic2, random_state = 1)
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


# In[55]:


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




