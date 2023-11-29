#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the csv file from a local location
df = pd.read_csv(r"C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\MANUSCRIPTS\FTC article\Data for Python scripts_FTC\FTC_ALL INTERACTION FIELDS.csv")
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


# # Split dataset into traning and test set

# In[19]:


# Split dataset into traning and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# # Feature selection using RFE and GridSearchCV

# In[20]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

pipe = Pipeline(steps=[("scaler", StandardScaler()), ('feat_sel', RFE(estimator= GradientBoostingRegressor(random_state=1), step=20)),
                       ('model', GradientBoostingRegressor(random_state=1))])
param_grid = {
                       # specify here the number of features you want to test using the RFE method
                       'feat_sel__n_features_to_select': [ 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                       # ... other parameters you want to search ...
                      }

feat_search = GridSearchCV(estimator=pipe,
                            param_grid=param_grid, scoring="neg_mean_squared_error",n_jobs= -1)
feat_search_cv = feat_search.fit(X_train, y_train)
feat_search_df = pd.DataFrame(feat_search_cv.cv_results_)
feat_search_df


# In[21]:


feat_search_cv.best_params_


# In[22]:


feat_search_cv.best_score_


# In[23]:


feat_search_cv.best_estimator_


# In[24]:


# Access the RFE step in the pipeline
rfe_estimator = feat_search_cv.best_estimator_.named_steps['feat_sel']

# Retrieve the selected features (returns a boolean array that indicates which features have been selected and eliminated)
selected_features = rfe_estimator.support_

# Get the feature names from the original feature set
all_features = X_train.columns.tolist()

# Filter the selected feature names
selected_feature_names = [feature for feature, selected in zip(all_features, selected_features) if selected]

# Transform X_train and X_test using the selected features
X_train_transformed = X_train[selected_feature_names]
X_test_transformed = X_test[selected_feature_names]


# In[25]:


selected_feature_names


# In[26]:


X_train = X_train_transformed
X_train


# In[27]:


X_test= X_test_transformed
X_test


# In[28]:


# Read the csv file from a local location
X_for_prediction = pd.read_csv(r"C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\MANUSCRIPTS\FTC article\Data for Python scripts_FTC\FTC_5 CoMSIA fields_14 Tryptophyllin L.csv")
X_for_prediction


# In[29]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

# Assuming 'X_for_prediction' is your DataFrame
selected_columns_df = X_for_prediction.loc[:, selected_feature_names].copy()

# For example, using the GradientBoostingRegressor with the scaled data
model_3 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500, subsample=0.5, max_depth=2, random_state=1)

# Create the pipeline with feature scaling and the model
pipeline_3 = Pipeline(steps=[('scaler', StandardScaler()), ('model', model_3)])

# Fit the pipeline and make predictions
y_pred2 = pipeline_3.fit(X_train, y_train).predict(selected_columns_df)


# In[30]:


y_pred2


# In[31]:


df_y_pred2=pd.DataFrame(y_pred2)
df_y_pred2.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\MANUSCRIPTS\FTC article\FTC article submission\Figures\All Supplementary and  model statistics\y_pred_14Tryptophyllin.csv')


# In[32]:


selected_columns_df 

