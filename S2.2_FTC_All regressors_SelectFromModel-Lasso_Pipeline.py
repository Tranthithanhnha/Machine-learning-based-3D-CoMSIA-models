#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the csv file from a local location, remember using this slash / not \
df = pd.read_csv("C:/Users/84981/Dropbox/PC/Documents/Documents_old/Nha research_2018/FTC PROJECT/INTERATION FIELDS_FTC_197 PEPTIDES/FTC_ALL INTERACTION FIELDS.csv")
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


# In[20]:


df_X_train= pd.DataFrame(X_train)
df_X_train


# In[21]:


df_X_test= pd.DataFrame(X_test)
df_X_test


# # Feature selection using SelectFromModel-Lasso

# In[22]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

estimator = Pipeline(steps=[('scaler', StandardScaler()), ('LassoCV', LassoCV(eps=0.001, n_alphas=100, max_iter = 100000,random_state=1, cv=5))])  #  Linear Regression with L1 regularization
sfm = SelectFromModel(estimator, threshold=0.01, importance_getter='named_steps.LassoCV.coef_') #the smaller the threshold the more features selected
selector = sfm.fit(X_train, y_train)
selected_features = selector.get_feature_names_out(input_features=None)
selected_features


# In[23]:


X_train = selector.transform(X_train)
X_test = selector.transform(X_test)


# In[24]:


df_X_train = pd.DataFrame(X_train)
df_X_train


# In[25]:


df_X_train.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\df_X_train_S2.2.csv')


# In[26]:


y_train.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_train_S2.2.csv')


# In[27]:


df_X_test = pd.DataFrame(X_test)
df_X_test


# In[28]:


df_X_test.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\df_X_test_S2.2.csv')


# In[29]:


y_test.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\y_test_S2.2.csv')


# # Import all regressors for model building

# In[30]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.preprocessing import StandardScaler

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import PassiveAggressiveRegressor, LinearRegression, Ridge,Lasso,LassoLars,ElasticNet,BayesianRidge, TweedieRegressor, SGDRegressor, HuberRegressor,QuantileRegressor   
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.svm import SVR, LinearSVR, NuSVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,HistGradientBoostingRegressor 
from xgboost import XGBRegressor 


# In[31]:


# Define a function that compares the CV perfromance of a set of predetrmined models 
def cv_comparison(models, X_train, y_train, cv):

    # Initiate a DataFrame for the averages and a list for all measures
    cv_accuracies = pd.DataFrame()

    # Loop through the models, run a CV, add the average scores to the DataFrame and the scores of 
    # all CVs to the list
    for model in models:
        r2s = []
        maes = []
        mses = []
        rmses = []
        # create pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectKBest
        from scipy.stats import spearmanr
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', model)])
                                
        r2 = np.round(cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=cv), 4)
        r2s.append(r2)
        r2_avg = round(r2.mean(), 4)
        std_r2 = np.std(r2)
        
        mae = -np.round(cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv), 4)
        maes.append(mae)
        mae_avg = round(mae.mean(), 4)
        std_mae = np.std(mae)
        # use np.round function for an array and only round for a number. In this case: mae is an array, mae.mean() is a number
        
        mse = -np.round(cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_squared_error', cv=cv), 4)
        mses.append(mse)
        mse_avg = round(mse.mean(), 4)
        std_mse = np.std(mse)
        
        rmse = -np.round(cross_val_score(pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv), 4)
        rmses.append(rmse)
        rmse_avg = round(rmse.mean(), 4)
        std_rmse = np.std(rmse)

                
        cv_accuracies[str(model)] = [r2s, r2_avg, std_r2, mae_avg,  std_mae, mse_avg,  std_mse, rmse_avg, std_rmse]
        cv_accuracies.index = ['R^2s', 'R^2_cv', 'Std_R2', 'MAE', 'Std_MAE','MSE', 'Std_MSE', 'RMSE', 'Std_RMSE']
      
    return cv_accuracies


# In[32]:


# Create the models to be tested
pls_reg = PLSRegression(n_components= 3)
mlr_reg = LinearRegression()
ElasticNet_reg = ElasticNet(random_state=1)
BayesianRidge_reg = BayesianRidge()
Tweedie_reg = TweedieRegressor()
Quantile_reg = QuantileRegressor()
GaussianProcess_reg = GaussianProcessRegressor(random_state=1)

SGDRegressor_reg = SGDRegressor()
Huber_reg = HuberRegressor()
ridge_reg = Ridge()
KernelRidge_reg = KernelRidge()
Lasso_reg = Lasso(alpha=0.1)
LassoLars_reg = LassoLars(alpha=.1, normalize=False)

svr_reg = SVR(kernel='rbf', epsilon=0.2)
linear_svr_reg = LinearSVR(random_state=1, tol=1e-05)
NuSVR_reg = svm.NuSVR(nu=0.1)
KNeighbors_reg = KNeighborsRegressor(n_neighbors=2)
MLP_reg = MLPRegressor()

DecisionTree_reg = DecisionTreeRegressor(random_state=1)
Bagging_reg = BaggingRegressor(base_estimator=SVR(),random_state=1)
rf_reg = RandomForestRegressor(random_state=1)
Ada_reg = AdaBoostRegressor(random_state=1, n_estimators=100)
gb_reg = GradientBoostingRegressor(random_state=1)
HGB_reg = HistGradientBoostingRegressor(random_state=1)


linear_xgb_reg = XGBRegressor(booster = 'gblinear', random_state=1, verbosity= 0)
xgb_reg = XGBRegressor(random_state=1, verbosity= 0)

### Put the models in a list to be used for Cross-Validation
models = [pls_reg, mlr_reg, ElasticNet_reg, BayesianRidge_reg,  Tweedie_reg,  Quantile_reg, GaussianProcess_reg,
            SGDRegressor_reg, Huber_reg, ridge_reg, KernelRidge_reg, Lasso_reg, LassoLars_reg,
            svr_reg, linear_svr_reg, NuSVR_reg, KNeighbors_reg, MLP_reg,
            DecisionTree_reg, Bagging_reg, rf_reg, Ada_reg,gb_reg, HGB_reg, linear_xgb_reg, xgb_reg]


### Run the Cross-Validation comparison with the models used in this analysis
cv_comp = cv_comparison(models, X_train, y_train, 5)
cv_comp.T


# In[33]:


# Export cv_comp.T as a csv.file with the name "cv_comp_T.csv" in FTC folder
cv_comp.T.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\cv_compare_S2.2.csv')


# In[ ]:





# In[34]:


# loop throught the models, calculate the following parameters for each regressor
model_parameters = pd.DataFrame()
for model in models:    
    # R2, MSE and RMSE of the training set
    y_pred_train = model.fit(X_train, y_train).predict(X_train)
    R2_train = r2_score(y_train,y_pred_train)
    MSE_train = mean_squared_error(y_train,y_pred_train)
    RMSE_train = np.sqrt(mean_squared_error(y_train,y_pred_train))
    # R2, MSE and RMSE of the test set
    y_pred_test = model.fit(X_train, y_train).predict(X_test)
    R2_test = r2_score(y_test,y_pred_test)
    MSE_test = mean_squared_error(y_test,y_pred_test)
    RMSE_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
    model_parameters[str(model)] = [R2_test, MSE_test, RMSE_test, R2_train, MSE_train, RMSE_train ]
    model_parameters.index = ['R2_test', 'MSE_test', 'RMSE_test', 'R2_train', 'MSE_train', 'RMSE_train']
model_parameters.T
    


# In[35]:


model_parameters.T.to_csv(r'C:\Users\84981\Dropbox\PC\Documents\Documents_old\Nha research_2018\FTC PROJECT\INTERATION FIELDS_FTC_197 PEPTIDES\model_parameters_S2.2.csv')

