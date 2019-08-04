#!/usr/bin/env python
# coding: utf-8

# Project Description:
# - Use same datasets as Project 1.
# - Preprocess data: Explore data and apply data scaling.
# 
# Regression Task:
# - Apply any two models with bagging and any two models with pasting.
# - Apply any two models with adaboost boosting
# - Apply one model with gradient boosting
# - Apply PCA on data and then apply all the models in project 1 again on data you get from PCA. Compare your results with results in project 2. You don't need to apply all the models twice. Just copy the result table from project 1, prepare similar table for all the models after PCA and compare both tables. Does PCA help in getting better results?
# - Apply deep learning models covered in class
# 
# Classification Task:
# - Apply two voting classifiers - one with hard voting and one with soft voting
# - Apply any two models with bagging and any two models with pasting.
# - Apply any two models with adaboost boosting
# - Apply one model with gradient boosting
# - Apply PCA on data and then apply all the models in project 1 again on data you get from PCA. Compare your results with results in project 1. You don't need to apply all the models twice. Just copy the result table from project 1, prepare similar table for all the models after PCA and compare both tables. Does PCA help in getting better results?
# - Apply deep learning models covered in class
# 
# Deliverables:
# - Use markdown to provide inline comments for this project.
# - Your outputs should be clearly executed in the notebook i.e. we should not need to rerun the code to obtain the outputs.
# - Visualization encouraged.
# - If you are submitting two different files, then please only one group member submit both the files. If you submit two files separately from different accounts, it will be submitted as two different attempts.
# - If you are submitting two different files, then please follow below naming convetion:
#     Project2_Regression_GroupXX_Firstname1_Firstname2.ipynb
#     Project2_Classification_GroupXX_Firstname1_Firstname2.ipynb
# - If you are submitting single file, then please follow below naming convetion:
#     Project2_Both_GroupXX_Firstname1_Firstname2.ipynb
# 
# Questions regarding the project:
# - We have created a discussion board under Projects folder on e-learning. Create threads over there and post your queries related to project there.
# - We will also answer queries there. We will not be answering any project related queries through the mail.

# ### Project - GROUP 19:
# #### Regression Tasks are performed in the following code: 
# #### 1. Bagging -  Linear Regression and Lasso 
# #### 2. Pasting - KNN, Ridge
# #### 3. Ada Boosting - Linear SVM, SVM kernel = rbf
# #### 4. Gradient Boosting - Decision Tree
# #### 5. Neural Networks
# #### 6. PCA 
# 
# 
# #### Bharat Banjade 
# #### Pavithra Gunasekaran

# ## Exploratory Data Analysis  

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_auditRisk = pd.read_csv("audit_risk.csv")
data_trial = pd.read_csv("trial.csv")
data_auditRisk.T


# In[3]:


data_auditRisk.rename(columns={'PROB': 'PROB1'}, inplace=True)


# In[4]:


print(data_auditRisk.columns)
print(data_trial.columns)


# ###### Observation 1: Detection_Risk is a constant value 
# Dropping the Detection Risk column as it has a variance of zero.

# In[5]:


data_auditRisk = data_auditRisk.drop("Detection_Risk", axis = 1)


# In[6]:


data_trial


# In[7]:


data_auditRisk.head(2)


# #### SCORE_A and SCORE_B in data_trial are 10* Score_A and 10*Score_B of data_auditRisk

# In[8]:


data_auditRisk["Score_A"] = data_auditRisk["Score_A"]*10
data_auditRisk["Score_B"] = data_auditRisk["Score_B"]*10


# #### If the Score_A and Score_B values are adjusted, all common columns except 'Risk' hold the same value with the df observations in the same sequence
# 
# Common columns:
# 
# c_cols = ['Sector_score', 'LOCATION_ID', 'PARA_A', 'Score_A', 'PARA_B',
#        'Score_B', 'TOTAL', 'numbers', 'Money_Value', 'History', 'Score', 'Risk']
#        
# Columns in data_trial but not in data_auditRisk:
# 
# only_in_trial_cols = ['Marks', 'MONEY_Marks', 'District', 'Loss', 'LOSS_SCORE', 'History_score']

# In[9]:



dfwith_risk = ['Sector_score', 'LOCATION_ID', 'PARA_A', 'Score_A', 'PARA_B', 'Score_B', 'TOTAL', 'numbers', 'Money_Value', 'History','Score', 'Risk']
dfwithout_risk = ['Sector_score', 'LOCATION_ID', 'PARA_A', 'Score_A', 'PARA_B', 'Score_B', 'TOTAL', 'numbers', 'Money_Value', 'History','Score']
dfwith_risk_upper = [x.upper() for x in dfwith_risk]
dfwithout_risk_upper = [x.upper() for x in dfwithout_risk]

audit_names = data_auditRisk.columns
audit_names_upper =  [x.upper() for x in audit_names]
data_auditRisk.columns = audit_names_upper

trial_names = data_trial.columns
trial_names_upper =  [x.upper() for x in trial_names]
data_trial.columns = trial_names_upper


# In[10]:


Mainrisk_df = data_auditRisk.merge(data_trial, on=dfwithout_risk_upper)
Mainrisk_df.shape
Mainrisk_df = Mainrisk_df.drop_duplicates()
Mainrisk_df.shape


# In[11]:


Mainrisk_df.columns


# In[12]:


Mainrisk_df['MONEY_VALUE'] = Mainrisk_df["MONEY_VALUE"].fillna(Mainrisk_df["MONEY_VALUE"].mean())

Mainrisk_df.isnull().any()


# In[13]:


Mainrisk_df = Mainrisk_df.copy()

# Check type conversions
Mainrisk_df.dtypes


# In[14]:


Mainrisk_df["LOCATION_ID"]= Mainrisk_df["LOCATION_ID"].replace("LOHARU", 45)
Mainrisk_df["LOCATION_ID"]= Mainrisk_df["LOCATION_ID"].replace("NUH", 46)
Mainrisk_df["LOCATION_ID"]= Mainrisk_df["LOCATION_ID"].replace("SAFIDON", 47)


# In[15]:


Mainrisk_df["LOCATION_ID"].unique()


# In[16]:


Mainrisk_df.describe()


# Here in the above description it is observed that some of the columns like PARA_B, TOTAL, RSIK_B are having the outliers as their respective maximum values are greater than the value of their 3rd quartile. Hence there are outliers present for these columns.

# In[17]:


plt.boxplot(Mainrisk_df['PARA_B'])


# In[18]:


Mainrisk_df[Mainrisk_df['PARA_B']==1264.630000]


# In[19]:


Mainrisk_df.shape


# In[20]:


df_outlier = Mainrisk_df[Mainrisk_df.PARA_B != 1264.630000]


# In[21]:


plt.boxplot(df_outlier['PARA_B'])


# In[22]:


df_outlier[['MONEY_VALUE','RISK_D']].describe()


# In[23]:


df_outlier[(df_outlier['INHERENT_RISK'] == 622.838000) | (df_outlier['TOTAL'] == 191.360000) | (df_outlier['MONEY_VALUE'] == 935.030000) |(df_outlier['RISK_D'] == 561.018000)]


# In[24]:


Mainfinal_df = df_outlier[df_outlier.PARA_B != 1264.630000] = df_outlier[df_outlier.PARA_B != 1264.630000] = df_outlier[(df_outlier['INHERENT_RISK'] != 622.838000) & (df_outlier['TOTAL'] != 191.360000) & (df_outlier['MONEY_VALUE'] != 935.030000) & (df_outlier['RISK_D'] != 561.018000)]


# In[25]:


Mainfinal_df


# In[26]:


plt.boxplot(Mainfinal_df['INHERENT_RISK'])


# In[27]:


Mainfinal_df['RISK'] = Mainfinal_df['RISK_x'] | Mainfinal_df['RISK_y']


# In[28]:


Mainfinal_df = Mainfinal_df.drop(['RISK_x','RISK_y'],axis=1)


# In[29]:


Mainfinal_df


# In[30]:


Mainfinal_df.describe()


# In[31]:


Mainfinal_df.info()


# In[32]:


Mainfinal_df = Mainfinal_df.drop(['MONEY_MARKS','DISTRICT'],axis=1)


# In[33]:


Mainfinal_df.columns


# ## Visualizations
# Plotting the Sector_score vs Risk
# 
# 
# Here it is observed that the Risk is 1 for the sector_score between 2.72 and 3.89

# In[34]:


sns.countplot(x='SECTOR_SCORE',data=Mainfinal_df[['SECTOR_SCORE','RISK']],
              hue="RISK").set_title("Sector_score Vs Risk")
plt.xticks(rotation=45)


# ### Plotting of Risk for location_id
# It can be observed that the risk is 1 for location with id 8,23,2, and 16

# In[35]:


fig = plt.figure(figsize=(20,20))
sns.countplot(x='LOCATION_ID',data=Mainfinal_df[['LOCATION_ID','RISK']],
              hue="RISK").set_title("LOCATION_ID Vs RISK")
plt.xticks(rotation=45)


# ### Plot for History vs Risk
# It is observed that for the zero history the risk is less i.e., risk is zero,

# In[36]:


fig = plt.figure(figsize=(20,20))
sns.countplot(x='HISTORY',data=Mainfinal_df[['HISTORY','RISK']],
              hue="RISK").set_title("HISTORY Vs RISK")
plt.xticks(rotation=45)


# ### Plot for District - Loss vs Risk
# It is observed that the District-loss =2 has less risk as risk=0. 

# In[37]:


fig = plt.figure(figsize=(20,20))
sns.countplot(x='DISTRICT_LOSS',data=Mainfinal_df[['DISTRICT_LOSS','RISK']],
              hue="RISK").set_title("DISTRICT_LOSS Vs RISK")
plt.xticks(rotation=45)


# ### Plot for numbers and risk
# Here numbers refers to number of transactions. Here the risk=0 for numbers of transactions =5.  

# In[38]:


fig = plt.figure(figsize=(20,20))
sns.countplot(x='NUMBERS',data=Mainfinal_df[['NUMBERS','RISK']],
              hue="RISK").set_title("NUMBERS Vs RISK")
plt.xticks(rotation=45)


# ### Plot for Risk percentage
# It is observed that the value_counts for risk=0 and risk=1 are in the same degree. So it can be said that there is no class imbalance problem. 

# In[39]:


##fig = plt.figure(figsize=(20,20))
sns.countplot(x='RISK',data=Mainfinal_df[['RISK']],
              hue="RISK").set_title(" NO RISK VS RISK")
plt.xticks(rotation=45)


# Here it is observed from the above plot that there is good linear-correlation between INHERENT_RISK and AUDIT_RISK when RISK=1
# 
# Here it is observed that the linear correlation between INHERENT_RISK and AUDIT_RISK when RISK=0 is not so good.
# 
# From the above plots it is said that, anything above INHERENT_RISK of 3.5 it is said that the risk is high which is 1.
# The distribution is also different for risk=0 and risk=1.
# 
# 

# ### Scaling the features and splitting the data into X and y. 

# The data is splitted into to_scale_x_df and y_final_reg which is the target.
# 
# The features scaling is performed using MinMaxScaler and StandardScaler as well. 

# In[40]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

Mainfinal_df1 = Mainfinal_df.copy()
mm_scaler = MinMaxScaler()
std_scaler = StandardScaler()

#y_final_reg = final_df['AUDIT_RISK']# Regression y
#y_final_clf = final_df['RISK'] # Classification y
y_regFinal = Mainfinal_df['AUDIT_RISK']# Regression y
y_clfFinal = Mainfinal_df['RISK'] # Classification y

#to_scale_x_df = final_df1.drop(["AUDIT_RISK","RISK"], axis =1)
scale_x_df = Mainfinal_df1.drop(["AUDIT_RISK","RISK"], axis =1)

#minmax_x_df = to_scale_x_df.copy()
#standard_x_df = to_scale_x_df.copy()
mm_x_df = scale_x_df.copy()
std_x_df = scale_x_df.copy()

num_cols = ['SECTOR_SCORE', 'LOCATION_ID','PARA_A', 'SCORE_A', 'RISK_A', 'PARA_B',
       'SCORE_B', 'RISK_B', 'TOTAL', 'NUMBERS', 'SCORE_B.1', 'RISK_C',
       'MONEY_VALUE', 'SCORE_MV', 'RISK_D', 'DISTRICT_LOSS', 'PROB1', 'RISK_E',
       'HISTORY','RISK_F', 'SCORE', 'INHERENT_RISK', 'CONTROL_RISK',
        'MARKS', 'LOSS','PROB', 'LOSS_SCORE', 'HISTORY_SCORE']
num_cols = [x.upper() for x in num_cols]

#minmax_x_df[num_cols] = minmax_scaler.fit_transform(minmax_x_df[num_cols])       # MinMax scaled X
#standard_x_df[num_cols] = standard_scaler.fit_transform(standard_x_df[num_cols])    # Std scaled X

mm_x_df[num_cols] = mm_scaler.fit_transform(mm_x_df[num_cols])       # MinMax scaled X
std_x_df[num_cols] = std_scaler.fit_transform(std_x_df[num_cols])    # Std scaled X


#X=minmax_x_df[num_cols]
#y=y_final_reg
X=mm_x_df[num_cols]
y=y_regFinal


# In[41]:


X.columns
X.shape


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

attributes = Mainfinal_df.columns[:6]
scatter_matrix(X[attributes], figsize = (15,15), c = y, alpha = 0.8, marker = 'O')


# ### Correlation matrix

# In[43]:


# heatmap
# Correlation matrix - linear relation among independent attributes and with the Target attribute

plt.figure(figsize = (25,25))
sns.heatmap(Mainfinal_df.corr(), square = True, linecolor = 'red', annot = True)
Mainfinal_df.shape


# ### Splitting the Data into three different sets: Training set, Testing set and the validation set 

# In[ ]:


# from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0


# # Bagging
# # 1. Linear Regression with GridSearchCV for the best Parameters

# In[176]:


from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV


lreg = LinearRegression()
n_estimators_vals = [100, 200, 300, 400, 500]
max_samples_vals = [10, 50, 70, 100, 120, 150, 170, 200]


param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals)

lreg_bag = BaggingRegressor(lreg,bootstrap = False, random_state=0)

grid_search = GridSearchCV(lreg_bag, param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals), cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[177]:


lreg_bag =  BaggingRegressor(lreg, n_estimators=200, max_samples=400, bootstrap=False, random_state=0)
lreg_bag.fit(X_trainval, y_trainval)
print(lreg)
print('Train score: %.4f'%lreg_bag.score(X_trainval, y_trainval))
print('Test score: %.4f'%lreg_bag.score(X_test, y_test))


predictions = lreg_bag.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
# calculate these metrics by hand!
from sklearn import metrics

print('mae:', metrics.mean_absolute_error(y_test, predictions))
print('mse:', metrics.mean_squared_error(y_test, predictions))
print('rmse:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[178]:



predictions = lreg_bag.predict(X_test)
#plot1
plt.scatter(y_test, predictions)
#plot 2: residual
plt.scatter(predictions, predictions - y_test, c = 'b')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

mse=metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

sns.distplot((y_test-predictions), bins=500)


# ### Regression Relationship 

# In[49]:



Mainfinal_df[['LOCATION_ID','RISK']] = Mainfinal_df[['LOCATION_ID','RISK']].astype('int') 
    
sns.pairplot(Mainfinal_df, x_vars=['SECTOR_SCORE','LOCATION_ID','PARA_A'], y_vars=["AUDIT_RISK"],
             height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['RISK_A', 'PARA_B','SCORE_B'], y_vars=["AUDIT_RISK"],
           height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['RISK_B', 'TOTAL', 'NUMBERS'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['SCORE_B.1', 'RISK_C','MONEY_VALUE'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6,plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['SCORE_MV', 'RISK_D', 'DISTRICT_LOSS'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['PROB1', 'RISK_E','HISTORY'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['PROB', 'RISK_F', 'SCORE'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['INHERENT_RISK','CONTROL_RISK', 'MARKS'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['LOSS'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})

sns.pairplot(Mainfinal_df, x_vars=['LOSS_SCORE','HISTORY_SCORE', 'RISK'], y_vars=["AUDIT_RISK"],
            height=4.2, aspect=1, kind="reg", size = 6, plot_kws={'line_kws':{'color':'red'}})


# # PASTING
# # 2. K-NN with GridSearchCV for the best parameters
# 

# In[50]:


from sklearn.neighbors import KNeighborsRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
train_score_array = []
test_score_array = []

knn_reg = KNeighborsRegressor(3)
knn_reg.fit(X_trainval, y_trainval)
train_score_array.append(knn_reg.score(X_trainval, y_trainval))
test_score_array.append(knn_reg.score(X_test, y_test))
print(train_score_array)
print(test_score_array)


# In[51]:


from sklearn import metrics
knn_tr_pred = knn_reg.predict(X_trainval)
knn_test_pred = knn_reg.predict(X_test)
knn_tr_mse = metrics.mean_squared_error(y_trainval, knn_tr_pred)
knn_tr_rmse = np.sqrt(knn_tr_mse)
knn_test_mse = metrics.mean_squared_error(y_test, knn_test_pred)
knn_test_rmse = np.sqrt(knn_test_mse)

print('train mse: ', knn_tr_mse)
print('train rmse: ', knn_tr_rmse)

print('\ntest mse: ', knn_test_mse)
print('test rmse: ', knn_test_rmse)

print('\ntrain score: ', knn_reg.score(X_trainval, y_trainval))
print('test score: ', knn_reg.score(X_test, y_test) )


# In[52]:


from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV


knn_reg = KNeighborsRegressor(3)
n_estimators_vals = [100, 200, 300, 400, 500]
max_samples_vals = [10, 50, 70, 100, 120, 150, 170, 200]


param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals)

knn_bag = BaggingRegressor(knn_reg,bootstrap = False, random_state=0) #pasting

grid_search = GridSearchCV(knn_bag, param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals), cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[53]:


from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(3)
bag_reg =  BaggingRegressor(knn_reg, n_estimators=200, max_samples=100, bootstrap=False, random_state=0)

bag_reg.fit(X_trainval, y_trainval)
y_pred = bag_reg.predict(X_test)


# In[54]:


bag_reg.fit(X_trainval, y_trainval)
print('Train score: {:.2f}'.format(bag_reg.score(X_trainval, y_trainval)))
print('Test score: {:.2f}'.format(bag_reg.score(X_test, y_test)))


# In[55]:


from sklearn import metrics
knn_tr_pred = bag_reg.predict(X_trainval)
knn_test_pred = bag_reg.predict(X_test)
knn_tr_mse = metrics.mean_squared_error(y_trainval, knn_tr_pred)
knn_tr_rmse = np.sqrt(knn_tr_mse)
knn_test_mse = metrics.mean_squared_error(y_test, knn_test_pred)
knn_test_rmse = np.sqrt(knn_test_mse)

print('train mse: ', knn_tr_mse)
print('train rmse: ', knn_tr_rmse)

print('\ntest mse: ', knn_test_mse)
print('test rmse: ', knn_test_rmse)


# # Pasting
# # 3. RIDGE with GridSearchCV for best params

# In[56]:


from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.001)
ridge.fit(X_trainval,y_trainval)
print('Train score: {:.4f}'.format(ridge.score(X_trainval,y_trainval)))
print('Test score: {:.4f}'.format(ridge.score(X_test, y_test)))


# In[57]:


ridge_reg = Ridge(alpha = 0.001)
n_estimators_vals = [100, 200, 300, 400, 500]
max_samples_vals = [10, 50, 70, 100, 120, 150, 170, 200]


param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals)

rid_bag = BaggingRegressor(ridge_reg,bootstrap = False, random_state=0)

grid_search = GridSearchCV(rid_bag, param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals), cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[58]:


ridge_reg = Ridge(alpha = 0.001)
from sklearn.ensemble import BaggingRegressor
bag_reg =  BaggingRegressor(ridge_reg, n_estimators=100, max_samples=50, bootstrap=False, random_state=0)

bag_reg.fit(X_trainval, y_trainval)
y_pred = bag_reg.predict(X_test)


# bag_reg.fit(X_trainval, y_trainval)
# print('Train score: {:.2f}'.format(bag_reg.score(X_trainval, y_trainval)))
# print('Test score: {:.2f}'.format(bag_reg.score(X_test, y_test)))

# In[59]:


from sklearn import  metrics
ridge_tr_pred = bag_reg.predict(X_trainval)
ridge_test_pred =bag_reg.predict(X_test)
ridge_tr_mse = metrics.mean_squared_error(y_trainval,ridge_tr_pred)
ridge_tr_rmse = np.sqrt(ridge_tr_mse)
ridge_test_mse = metrics.mean_squared_error(y_test, ridge_test_pred)
ridge_test_rmse = np.sqrt(ridge_test_mse)

print('train mse: ', ridge_tr_mse)
print('train rmse: ', ridge_tr_rmse)

print('test mse: ', ridge_test_mse)
print('test rmse: ', ridge_test_rmse)


# # Bagging 
# # 3. Lasso with GridSearchCV for best parameters

# In[60]:


## lasso
from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.001)
lasso.fit(X_trainval,y_trainval)
print('Train score: {:.4f}'.format(lasso.score(X_trainval,y_trainval)))
print('Test score: {:.4f}'.format(lasso.score(X_test, y_test)))


# In[61]:


lasso_reg = Lasso(alpha = 0.001)
n_estimators_vals = [100, 200, 300, 400, 500]
max_samples_vals = [10, 50, 70, 100, 120, 150, 170, 200]


param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals)

lass_bag = BaggingRegressor(lasso_reg,bootstrap = True, random_state=0)

grid_search = GridSearchCV(lass_bag, param_grid = dict(n_estimators=n_estimators_vals, max_samples = max_samples_vals), cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)


# In[136]:


print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[62]:


lasso_reg = Lasso(alpha = 0.001)
bag_reg =  BaggingRegressor(lasso_reg, n_estimators=200, max_samples=50, bootstrap=True, random_state=0)

bag_reg.fit(X_trainval, y_trainval)


# In[63]:


bag_reg.fit(X_trainval, y_trainval)
print('Train score: {:.2f}'.format(bag_reg.score(X_trainval, y_trainval)))
print('Test score: {:.2f}'.format(bag_reg.score(X_test, y_test)))


# In[64]:


from sklearn import  metrics
lasso_tr_pred = bag_reg.predict(X_trainval)
lasso_test_pred =bag_reg.predict(X_test)
lasso_tr_mse = metrics.mean_squared_error(y_trainval,lasso_tr_pred)
lasso_tr_rmse = np.sqrt(lasso_tr_mse)
lasso_test_mse = metrics.mean_squared_error(y_test, lasso_test_pred)
lasso_test_rmse = np.sqrt(lasso_test_mse)

print('train mse: ', lasso_tr_mse)
print('train rmse: ', lasso_tr_rmse)

print('test mse: ', lasso_test_mse)
print('test rmse: ', lasso_test_rmse)


# ## 4. SVM with kernel = Linear 

# In[148]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np


# In[149]:


from sklearn import svm
svm_r = svm.SVR(kernel='linear', C = 100)
svm_r.fit(X_trainval, y_trainval)

svmr_tr_pred = svm_r.predict(X_trainval)
svmr_test_pred = svm_r.predict(X_test)
print('Train Score:',svm_r.score(X_trainval,y_trainval))      
print('Train Score:',svm_r.score(X_test, y_test))


svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)

print('\ntrain mse: ', svm_tr_mse)
print('train rmse: ', svm_tr_rmse)

print('test mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)
print('Train Score:',svm_r.score(X_trainval,y_trainval))      
print('Train Score:',svm_r.score(X_test, y_test))


# # 5. Ada Boosting on Linear SVM
# 
# #### After using GRidSearchCV on Linear SVM, we get best parameters used to fit the model. We get 'learning rate' = 0.01 and 'n_estimators' = 100, with a best cross validation score of 0.55.
# 
# #### We Perform Ada Boost on the new model with the best parameters which results in a Train Score = 0.49 , Test Score = 0.40.
# 
# #### TRAIN RMSE = 11.162 
# #### TEST RMSE = 15.98
# 

# In[150]:


from sklearn.ensemble import AdaBoostRegressor
svm_r = svm.SVR(kernel='linear', C = 100)

n_estimators_vals = [100, 200, 300, 400, 500]
learning_rate_vals = [0.01, 0.1, 0.3, 0.5, 1.0]


param_grid = dict(n_estimators=n_estimators_vals, learning_rate = learning_rate_vals)
svm_r_bag = AdaBoostRegressor(svm_r, random_state=0)
grid_search = GridSearchCV(svm_r_bag, param_grid = dict(n_estimators=n_estimators_vals, learning_rate = learning_rate_vals), cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[151]:


svm_r = svm.SVR(kernel='linear', C = 100)
ada_reg =  AdaBoostRegressor(svm_r, n_estimators=100, learning_rate=0.01, random_state=0)

ada_reg.fit(X_trainval, y_trainval)
y_pred = ada_reg.predict(X_test)


# In[152]:


ada_reg.fit(X_trainval, y_trainval)
print('Train score: {:.2f}'.format(ada_reg.score(X_trainval, y_trainval)))
print('Test score: {:.2f}'.format(ada_reg.score(X_test, y_test)))


# In[153]:


svmr_tr_pred = ada_reg.predict(X_trainval)
svmr_test_pred = ada_reg.predict(X_test)
print('Train Score:',ada_reg.score(X_trainval,y_trainval))      
print('Train Score:',ada_reg.score(X_test, y_test))

svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)

print('train mse: ', svm_tr_mse)
print('test rmse: ', svm_tr_rmse)

print('\ntest mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)
print('Train Score:',ada_reg.score(X_trainval,y_trainval))      
print('Test Score:',ada_reg.score(X_test, y_test))


# ## 6. SVM with kernel = rbf - Ada Boosting
# 
# #### After using GRidSearchCV on SVM Kernel = rbf, we get best parameters used to fit the model. We get 'learning rate' = 0.01 and 'n_estimators' = 100, with a best cross validation score of 0.78.
# 
# #### We Perform Ada Boost on the new model with the best parameters which results in a Train Score=0.75 , Test Score = 0.58
# 
# 

# In[71]:


from sklearn.ensemble import AdaBoostRegressor
svm_r = svm.SVR(kernel='rbf', C = 100,gamma=0.1)

n_estimators_vals = [100, 200, 300, 400, 500]
learning_rate_vals = [0.01, 0.1, 0.3, 0.5, 1.0]


param_grid = dict(n_estimators=n_estimators_vals, learning_rate = learning_rate_vals)
svm_r_bag = AdaBoostRegressor(svm_r, random_state=0)
grid_search = GridSearchCV(svm_r_bag, param_grid = dict(n_estimators=n_estimators_vals, learning_rate = learning_rate_vals), cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[72]:


svm_r = svm.SVR(kernel='rbf', C = 100,gamma=0.1)
ada_reg_rbf =  AdaBoostRegressor(svm_r, n_estimators=100, learning_rate=0.01, random_state=0)

ada_reg_rbf.fit(X_trainval, y_trainval)
y_pred = ada_reg_rbf.predict(X_test)


# In[73]:


ada_reg_rbf.fit(X_trainval, y_trainval)
print('Train score: {:.2f}'.format(ada_reg.score(X_trainval, y_trainval)))
print('Test score: {:.2f}'.format(ada_reg.score(X_test, y_test)))


# ## 7. Decision Tree with gradient boosting
# 
# #### After using GRidSearchCV on Decision Tree Regressor, we get best parameters used to fit the model. We get 'Max Depth' = 8  with a best cross validation score of 0.65.
# 
# #### We Perform GRADIENT BOOSTING on the new model with the best parameters which results in an accuracy of 0.734 on the TEST set. 
# 
# 
# 

# In[74]:



from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score , GridSearchCV

dt = DecisionTreeRegressor()
max_depth_vals=[4,6,8,10]
param_grid = dict(max_depth = max_depth_vals)

gs_dt = GridSearchCV(dt, param_grid = dict(max_depth = max_depth_vals), return_train_score=True)


gs_dt
gs_dt.fit(X_trainval, y_trainval)
print("Best parameters: {}".format(gs_dt.best_params_))
print("Best cross-validation score: {:.2f}".format(gs_dt.best_score_))


# In[75]:


from  sklearn.ensemble import GradientBoostingRegressor


# In[76]:


dt_r = DecisionTreeRegressor(max_depth=8)
n_estimators_vals = [1,2,3,4,5]
learning_rate_vals = [0.01, 0.1, 0.3, 0.5, 1.0]


param_grid = dict(n_estimators=n_estimators_vals, learning_rate = learning_rate_vals)
gbrt = GradientBoostingRegressor(dt_r , random_state=0)
gs_dt_r = GridSearchCV(gbrt, param_grid = dict(n_estimators=n_estimators_vals, learning_rate = learning_rate_vals), cv=10, return_train_score=True)
grid_search


# In[77]:


from  sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=8, n_estimators=100, learning_rate=1.0, random_state=42)
gbrt.fit(X_trainval, y_trainval)


# In[78]:


print("Accuracy on training set: {:.3f}".format(gbrt.score(X_trainval, y_trainval)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# ## DEEP NEURAL NETWORKS

# In[126]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[130]:


#step 1: build model
model1 = Sequential()
#input layer
model1.add(Dense(10, input_dim = 28, activation = 'relu'))
#hidden layers
#output layer
model1.add(Dense(1, activation = 'sigmoid'))

#step 2: make computational graph - compile
model1.compile(loss= 'mse' , optimizer = 'adam',metrics = ['mse'] )

#step 3: train the model - fit
model1.fit(X_train, y_train, epochs = 10, batch_size = 400)


# In[132]:



model1.evaluate(X_train, y_train)


# In[135]:


model1.evaluate(X_valid, y_valid)


# ## PCA
# 
# The reduced data set has only 10 features after using PCA for all the models as specified.

# In[79]:


## PCA
from sklearn.decomposition import PCA



pca = PCA().fit(X)


# In[80]:


# split data into train+validation set and test set
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0


# In[81]:


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.show()


# In[82]:


pca = PCA(n_components=10)

X_trainval = pca.fit_transform(X_trainval)
X_test = pca.transform(X_test)


# In[83]:


X_trainval.shape


# In[84]:


X_test.shape


# In[85]:


pca.explained_variance_


# In[86]:


pca.n_components_


# In[87]:


pca.explained_variance_ratio_


# In[88]:


np.sum(pca.explained_variance_ratio_)


# ## 1. K-NN after PCA
# 
# After using GRidSearchCV on KNN, we get best parameters used to fit the model. We get 'n_neighbors' = 12 and 'weights' = distance, with a best cross validation score of 0.55.
# 
# We Perform Ada Boost on the new model with the best parameters which results in a Train Score=0.515 , Test Score = 0.400
# 
# 
# 

# In[89]:


## K-NN
from sklearn.neighbors import KNeighborsRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
train_score_array = []
test_score_array = []

for k in range(1,10):
    knn_reg = KNeighborsRegressor(k)
    knn_reg.fit(X_trainval, y_trainval)
    train_score_array.append(knn_reg.score(X_trainval, y_trainval))
    test_score_array.append(knn_reg.score(X_test, y_test))

x_axis = range(1,10)
plt.plot(x_axis, train_score_array, c = 'g', label = 'Train Score')
plt.plot(x_axis, test_score_array, c = 'b', label = 'Test Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('MSE')


# In[90]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()

from sklearn.model_selection import GridSearchCV
#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])
k_range = [1,3,5,7,9,12,15,17,20]          
weights_range = ['uniform','distance'] 
param_grid = dict(n_neighbors=k_range, weights = weights_range)


#grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[91]:



get_ipython().run_line_magic('matplotlib', 'inline')
train_score_array = []
test_score_array = []

knn_reg = KNeighborsRegressor(12)
knn_reg.fit(X_trainval, y_trainval)
train_score_array.append(knn_reg.score(X_trainval, y_trainval))
test_score_array.append(knn_reg.score(X_test, y_test))
print(train_score_array)
print(test_score_array)


# In[92]:


from sklearn import metrics
knn_tr_pred = knn_reg.predict(X_trainval)
knn_test_pred = knn_reg.predict(X_test)
knn_tr_mse = metrics.mean_squared_error(y_trainval, knn_tr_pred)
knn_tr_rmse = np.sqrt(knn_tr_mse)
knn_test_mse = metrics.mean_squared_error(y_test, knn_test_pred)
knn_test_rmse = np.sqrt(knn_test_mse)

print('train mse: ', knn_tr_mse)
print('train rmse: ', knn_tr_rmse)

print('test mse: ', knn_test_mse)
print('test rmse: ', knn_test_rmse)

print('train score: ', knn_reg.score(X_trainval, y_trainval))
print('test score: ', knn_reg.score(X_test, y_test) )


# ## 2. RIDGE after PCA
# 
# After using Cross validation on Ridge, we get best parameters used to fit the model. We get 'alpha' = 0.001, with a best cross validation score of 0.50 and a best TEST set score with best parameters of = 0.54.
# 
# The model results in a Train Score=0.5637 , Test Score = 0.5407

# In[93]:


from  sklearn.linear_model import Ridge

x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_trainval,y_trainval)
    train_score_list.append(ridge.score(X_trainval,y_trainval))
    test_score_list.append(ridge.score(X_test, y_test))


# In[94]:


print(train_score_list)
print(test_score_list)


# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

x_range1 = np.linspace(0.001, 1, 100).reshape(-1,1)
x_range2 = np.linspace(1, 10000, 10000).reshape(-1,1)

x_range = np.append(x_range1, x_range2)
coeff = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_trainval,y_trainval)
    coeff.append(ridge.coef_ )
    
coeff = np.array(coeff)

for i in range(0,10):
    plt.plot(x_range, coeff[:,i], label = 'feature {:d}'.format(i))

plt.axhline(y=0, xmin=0.001, xmax=9999, linewidth=1, c ='gray')
plt.xlabel(r'$\alpha$')
plt.xscale('log')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=3, fancybox=True, shadow=True)
plt.show()


# In[96]:


from sklearn.linear_model import Ridge
import numpy as np

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    
        ridge =Ridge()
        scores = cross_val_score(ridge, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'alpha': alpha}
            

ridge = Ridge(**best_parameters)
ridge.fit(X_trainval, y_trainval)
test_score = ridge.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)
print("Test set score with best parameters: {:.2f}".format(test_score))


# In[97]:


ridge = Ridge(alpha = 0.001)
ridge.fit(X_trainval,y_trainval)
print('Train score: {:.4f}'.format(ridge.score(X_trainval,y_trainval)))
print('Test score: {:.4f}'.format(ridge.score(X_test, y_test)))


# In[98]:


from sklearn import  metrics
ridge_tr_pred = ridge.predict(X_trainval)
ridge_test_pred =ridge.predict(X_test)
ridge_tr_mse = metrics.mean_squared_error(y_trainval,ridge_tr_pred)
ridge_tr_rmse = np.sqrt(ridge_tr_mse)
ridge_test_mse = metrics.mean_squared_error(y_test, ridge_test_pred)
ridge_test_rmse = np.sqrt(ridge_test_mse)

print('train mse: ', ridge_tr_mse)
print('train rmse: ', ridge_tr_rmse)

print('test mse: ', ridge_test_mse)
print('test rmse: ', ridge_test_rmse)


# ## 3. Lasso after PCA
# 
# After using Cross validation on Ridge, we get best parameters used to fit the model. We get 'alpha' = 0.001, with a best cross validation score of 0.50 and a best TEST set score with best parameters of = 0.54.
# 
# The model results in a Train Score=0.5637 , Test Score = 0.5406
# 

# In[99]:


from sklearn.linear_model import Lasso
x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    lasso = Lasso(alpha)
    lasso.fit(X_trainval,y_trainval)
    train_score_list.append(lasso.score(X_trainval,y_trainval))
    test_score_list.append(lasso.score(X_test, y_test))


# In[100]:


plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')


# In[101]:


get_ipython().run_line_magic('matplotlib', 'inline')

x_range1 = np.linspace(0.001, 1, 1000).reshape(-1,1)
x_range2 = np.linspace(1, 1000, 1000).reshape(-1,1)

x_range = np.append(x_range1, x_range2)
coeff = []

for alpha in x_range: 
    lasso = Lasso(alpha)
    lasso.fit(X_trainval,y_trainval)
    coeff.append(lasso.coef_ )
    
coeff = np.array(coeff)

for i in range(0,10):
    plt.plot(x_range, coeff[:,i], label = 'feature {:d}'.format(i))

plt.axhline(y=0, xmin=0.001, xmax=9999, linewidth=1, c ='gray')
plt.xlabel(r'$\alpha$')
plt.xscale('log')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=3, fancybox=True, shadow=True)
plt.show()


# In[102]:


from sklearn.linear_model import Lasso
import numpy as np

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:   
        lasso =Lasso()
        # perform cross-validation
        scores = cross_val_score(ridge, X_trainval, y_trainval, cv=5)
        # compute mean cross-validation accuracy
        score = np.mean(scores)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'alpha': alpha}
            
# rebuild a model on the combined training and validation set
lasso = Lasso(**best_parameters)
lasso.fit(X_trainval, y_trainval)
test_score = lasso.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)
print("Test set score with best parameters: {:.2f}".format(test_score))


# In[103]:


lasso = Lasso(alpha = 0.001)
lasso.fit(X_trainval,y_trainval)
print('Train score: {:.4f}'.format(lasso.score(X_trainval,y_trainval)))
print('Test score: {:.4f}'.format(lasso.score(X_test, y_test)))


# In[104]:


from sklearn import  metrics
lasso_tr_pred = lasso.predict(X_trainval)
lasso_test_pred =lasso.predict(X_test)
lasso_tr_mse = metrics.mean_squared_error(y_trainval,lasso_tr_pred)
lasso_tr_rmse = np.sqrt(lasso_tr_mse)
lasso_test_mse = metrics.mean_squared_error(y_test, lasso_test_pred)
lasso_test_rmse = np.sqrt(lasso_test_mse)

print('train mse: ', lasso_tr_mse)
print('train rmse: ', lasso_tr_rmse)

print('test mse: ', lasso_test_mse)
print('test rmse: ', lasso_test_rmse)


# ## 3. POLYNOMIAL after PCA
# 
# After using PCA data set, we fit the model to Polynomial Regression. We get a 'Traine RMSE' = 5.16 and a 'TEST RMSE' = 27.71. 
# 
# 
# 
# The model results in a Train Score=[0.5637408656641263, 0.8903515490291977] , Test Score = [0.5406840435699394, -0.8012171878312973]

# In[105]:


from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

train_score_list = []
test_score_list = []

for n in range(1,3):
    poly = PolynomialFeatures(n)
    X_train_poly = poly.fit_transform(X_trainval)
    X_test_poly = poly.transform(X_test) 
    lreg.fit(X_train_poly, y_trainval)
    train_score_list.append(lreg.score(X_train_poly, y_trainval))
    test_score_list.append(lreg.score(X_test_poly, y_test))


# In[106]:


print(train_score_list)
print(test_score_list)


# In[107]:


poly_train_pred = lreg.predict(X_train_poly)
poly_test_pred =lreg.predict(X_test_poly)
poly_mse = metrics.mean_squared_error(y_trainval,poly_train_pred)
poly_rmse = np.sqrt(poly_mse)
poly_test_mse = metrics.mean_squared_error(y_test, poly_test_pred)
poly_test_rmse = np.sqrt(poly_test_mse)

print('train mse: ', poly_mse)
print('train rmse: ', poly_rmse)

print('test mse: ', poly_test_mse)
print('test rmse: ', poly_test_rmse)



print(train_score_list)
print(test_score_list)


# ## 4. DECISION TREE after PCA
# 
# After using GRidSearchCV on the reduced PCA data set using Decision Tree Regressor, we get best parameters used to fit the model. We get 'Max Depth' = 8 with a best cross validation score of 0.65.
# 
# We fit the model with the best parameters which results in a TRIAN SCORE = 0.99797 and a TEST SCORE = 0.21358.
# 
# The TRAIN RMSE =10.309
# The TEST RMSE = 13.995 

# In[108]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score , GridSearchCV

DT_r = DecisionTreeRegressor()
max_depth_vals=[4,6,8,10]
param_grid = dict(max_depth = max_depth_vals)

DT_r = GridSearchCV(dt, param_grid = dict(max_depth = max_depth_vals), return_train_score=True)


DT_r
DT_r.fit(X_trainval, y_trainval)
print("Best parameters: {}".format(gs_dt.best_params_))
print("Best cross-validation score: {:.2f}".format(gs_dt.best_score_))


# In[109]:



DT_tr_pred = DT_r.predict(X_trainval)
DT_test_pred = DT_r.predict(X_test)
lreg = LinearRegression().fit(X_trainval, y_trainval)

pred_lr = lreg.predict(X_trainval)
pred_test =lreg.predict(X_test)


# In[110]:


pred_lr 
pred_test


# In[111]:


print('Train Score:',DT_r.score(X_trainval,y_trainval))      
print('Test Score:',DT_r.score(X_test, y_test))


# In[112]:


from sklearn import  metrics
pred_lr = lreg.predict(X_trainval)
pred_test =lreg.predict(X_test)
pred_lr_mse = metrics.mean_squared_error(y_trainval,pred_lr)
pred_lr_rmse = np.sqrt(pred_lr_mse)
pred_test_mse = metrics.mean_squared_error(y_test, pred_test)
pred_test_rmse = np.sqrt(pred_test_mse)

print('train mse: ', pred_lr_mse)
print('train rmse: ', pred_lr_rmse)

print('\ntest mse: ', pred_test_mse)
print('test rmse: ', pred_test_rmse)


# ## 5. SVM after PCA
# 
# After using GRidSearchCV on the reduced PCA data set using SVM, we get best parameters used to fit the model. We get 'C' = 100 with a best cross validation score of 0.65.
# 
# We fit the model with the best parameters which results in a TRIAN SCORE = 0.4416 and a TEST SCORE = 0.336.
# 
# The TRAIN RMSE = 11.663
# The TEST RMSE = 16.825 

# In[113]:


from sklearn.model_selection import GridSearchCV
#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])

from sklearn import svm
from sklearn.svm import SVR
import numpy as np

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
 
svm_r = svm.SVR()
grid_search = GridSearchCV(svm_r, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[114]:


# SVM Linear


from sklearn import svm
svm_r = svm.SVR(kernel='linear', C = 100)
svm_r.fit(X_trainval, y_trainval)

svmr_tr_pred = svm_r.predict(X_trainval)
svmr_test_pred = svm_r.predict(X_test)
print('Train Score:',svm_r.score(X_trainval,y_trainval))      
print('Test Score:',svm_r.score(X_test, y_test))





svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)

print('train mse: ', svm_tr_mse)
print('train rmse: ', svm_tr_rmse)

print('test mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)
print('Train Score:',svm_r.score(X_trainval,y_trainval))      
print('Test Score:',svm_r.score(X_test, y_test))


# In[115]:


from sklearn.model_selection import GridSearchCV
#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])

from sklearn import svm
from sklearn.svm import SVR
import numpy as np

param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100],'C': [0.001, 0.01, 0.1, 1, 10, 100]}
 
svm_r = svm.SVR()
grid_search = GridSearchCV(svm_r, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[116]:


svm_r = svm.SVR(kernel='rbf', C = 100,gamma=0.1)
svm_r.fit(X_trainval, y_trainval)

svmr_tr_pred = svm_r.predict(X_trainval)
svmr_test_pred = svm_r.predict(X_test)

svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)

print('train mse: ', svm_tr_mse)
print('train rmse: ', svm_tr_rmse)

print('test mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)
print('Train Score:',svm_r.score(X_trainval,y_trainval))      
print('Train Score:',svm_r.score(X_test, y_test))


# In[117]:


mse = svm_test_mse
rmse = svm_test_rmse


# ## 6. SVM with kernel = poly  after PCA
# 
# After using GRidSearchCV on the reduced PCA data set using SVM, we get best parameters used to fit the model. We get 'C' = 100, degree=1 with a best cross validation score of 0.65.
# 
# We fit the model with the best parameters which results in a TRIAN SCORE = 0.43365 and a TEST SCORE = 0.3276.
# 
# The TRAIN RMSE = 11.7464
# The TEST RMSE = 16.933 

# In[118]:


from sklearn.model_selection import GridSearchCV
#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])

from sklearn import svm
from sklearn.svm import SVR
import numpy as np

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],'degree': [1,2,3]}
 
svm_r = svm.SVR()
grid_search = GridSearchCV(svm_r, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)


# In[125]:


print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[119]:


svm_r = svm.SVR(kernel='poly', C = 100, degree=1)
svm_r.fit(X_trainval, y_trainval)

svmr_tr_pred = svm_r.predict(X_trainval)
svmr_test_pred = svm_r.predict(X_test)

svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)
print('Train Score:',svm_r.score(X_trainval,y_trainval))      
print('Test Score:',svm_r.score(X_test, y_test))

print('train mse: ', svm_tr_mse)
print('train rmse: ', svm_tr_rmse)

print('test mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)


# ## 7. RANDOM FOREST after PCA
# 
# After using GRidSearchCV on the reduced PCA data set using Random forest, we get best parameters used to fit the model. We get 'max_features': 20, 'n_estimators': 50 with a best cross validation score of 0.8640.
# 
# Since we are working on the REDUCED DATA SET, we have MAX_FEATURES = 10 which is LESS than the estimated best Parameter. 
# 
# THUS,We fit the model with the best parameters having MAX_FEATURES = 10 which results in a TRIAN SCORE = 0.92011 and a TEST SCORE = 0.33023.
# 
# TRAIN RMSE = 4.478,  TEST RMSE = 16.150

# In[163]:


from sklearn.ensemble import RandomForestRegressor

estimator = [20,50,70]
max_features_val= [10,15,20]

param_grid = dict(n_estimators=estimator, max_features=max_features_val)
print(param_grid)


# In[164]:


randomforest = RandomForestRegressor()

rfgs = GridSearchCV(randomforest, param_grid = param_grid, cv=10, scoring='r2')
rfgs.fit(X,y)


# In[165]:


rfgs.best_score_


# In[166]:


rfgs.best_params_


# In[167]:


randomforest_best = RandomForestRegressor(n_estimators= 50,max_features= 20 )


# ###### Here, we observe that the max_features = 20 is the best parameter which is LESS than the number of features available in our REDUCED DATA SET AFTER USING PCA
# 
# #### Thus, Taking max_features = 10, we get: 
# #### TRAIN SCORE = 0.92011
# #### TEST SCORE = 0.33023

# In[171]:


randomforest_best = RandomForestRegressor(n_estimators= 50,max_features= 10 )


# In[173]:


randomforest_best.fit(X_trainval,y_trainval)
rf_train_pred = randomforest_best.predict(X_trainval)
rf_test_pred = randomforest_best.predict(X_test)

print("train score for Random Forest Reg:",randomforest_best.score(X_trainval,y_trainval))
print("test score for Random Forest Reg:",randomforest_best.score(X_test,y_test))

rf_tr_mse = metrics.mean_squared_error(y_trainval, rf_train_pred)
rf_tr_rmse = np.sqrt(rf_tr_mse)
rf_test_mse = metrics.mean_squared_error(y_test, rf_test_pred)
rf_test_rmse = np.sqrt(rf_test_mse)

print('\ntrain mse: ', rf_tr_mse)
print('train rmse: ', rf_tr_rmse)

print('test mse: ', rf_test_mse)
print('test rmse: ', rf_test_rmse)


# # SUMMARY 

# #### This Project started off with basic vizualization and in detail EDA. We have considered validation set for more error free model.
# 
# #### This File Contains only the REGRESSION part of PROJECT2.
# 
# #### 1. We have implemented BAGGING on two models: LINEAR REGRESSION and LASSO. Both the models use GridSearchCV to find the best parameters. Linear Regression TEST RMSE = 14.04 while Lasso TEST RMSE = 12.71. These are the results obtained after performing Bagging with GridsearchCV
# 
# #### 2. We have implemented PASTING on two models: KNearestNeighbors and RIDGE. Both these models use GridSearchCV to find the best parameters.KNN has TEST RMSE = 16.584 while Ridge has a TEST RMSE = 11.76180
# 
# #### 3. We have implemented ADA BOOSTING on two models: Linear SVM and SVM with kernel RBF
# 
# #### 4. We have implemented GRADIENT BOOSTING on only Decision Tree.
# 
# #### For the neural networks, we donot have any hidden layers. 
# 
# #### PCA for all the models with the respective ROCs and AUCs have been obtained.
# 
