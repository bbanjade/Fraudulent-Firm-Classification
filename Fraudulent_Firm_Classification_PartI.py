#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


data_auditRisk = pd.read_csv("audit_risk.csv")
data_trial = pd.read_csv("trial.csv")
data_auditRisk.T

data_auditRisk.info()
data_trial.info()

data_auditRisk.head()
data_trial.head()


# In[4]:


data_auditRisk.rename(columns={'PROB': 'PROB1'}, inplace=True)


# In[5]:


print(data_auditRisk.columns)
print(data_trial.columns)


# In[6]:


## Detection_Risk is a constant value with a variance of zero, so drop it.
data_auditRisk = data_auditRisk.drop("Detection_Risk", axis = 1)


# In[7]:


data_trial.T
data_trial['Risk'].unique()


# In[8]:


## SCORE_A and SCORE_B in data_trial are 10 times of Score_A and Score_B of data_auditRisk
## If we adjust Score_A and Score_B values, all columns except 'Risk' hold the same value with the 
## df observations in the same sequence.

# Common columns:
# c_cols = ['Sector_score', 'LOCATION_ID', 'PARA_A', 'Score_A', 'PARA_B', 'Score_B', 'TOTAL', 'numbers', 'Money_Value', 'History', 'Score', 'Risk']


# Columns in trial_df but not in audit_risk_df:
# only_in_trial_cols = ['Marks', 'MONEY_Marks', 'District', 'Loss', 'LOSS_SCORE', 'History_score']

data_auditRisk["Score_A"] = data_auditRisk["Score_A"]*10
data_auditRisk["Score_B"] = data_auditRisk["Score_B"]*10


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


# c_with_risk_cols will result in an inner merge (~580 observations on dropping duplicates)
# c_without_risk_cols will result in 763 observations after dropping duplicates but with two target variables which can be reduced using a Logical OR in case 

Mainrisk_df = data_auditRisk.merge(data_trial, on=dfwithout_risk_upper)
Mainrisk_df.shape
Mainrisk_df = Mainrisk_df.drop_duplicates()
Mainrisk_df.shape


# In[11]:


Mainrisk_df.columns


# In[12]:


## Replacing NULL values with Central Imputation
## Only MONEY_VALUE HAD NAN value, and is replaced by mean value.
Mainrisk_df['MONEY_VALUE'] = Mainrisk_df["MONEY_VALUE"].fillna(Mainrisk_df["MONEY_VALUE"].mean())

Mainrisk_df.isnull().sum()


# In[13]:


## changing LOCATION_ID from object to categorical and then changing the string values to unique numbers
Mainrisk_df = Mainrisk_df.copy()
Mainrisk_df[['LOCATION_ID']] = Mainrisk_df[['LOCATION_ID']].astype('category')

Mainrisk_df["LOCATION_ID"]= Mainrisk_df["LOCATION_ID"].replace("LOHARU", 45)
Mainrisk_df["LOCATION_ID"]= Mainrisk_df["LOCATION_ID"].replace("NUH", 46)
Mainrisk_df["LOCATION_ID"]= Mainrisk_df["LOCATION_ID"].replace("SAFIDON", 47)
Mainrisk_df["LOCATION_ID"].unique()


# In[14]:


# Removing Outliers
Mainrisk_df.describe()


# In[ ]:


## PARA_B, TOTAL, RISK_B has outliers due to higher maximum values than the 3rd quartile. 
## target column, for regression, is not touched.


# In[15]:


plt.boxplot(Mainrisk_df['AUDIT_RISK'])


# In[16]:


plt.boxplot(Mainrisk_df['PARA_B'])


# In[17]:


## PARA_B got only one observation, which is outlier.
Mainrisk_df[Mainrisk_df['PARA_B']==1264.630000]

## Removing the outlier for the PARA_B column
Mainrisk_df_rmout = Mainrisk_df[Mainrisk_df.PARA_B != 1264.630000]


# In[18]:


## following boxplot shows removal of outliers for PARA_B
plt.boxplot(Mainrisk_df_rmout['PARA_B'])


# In[19]:


Mainrisk_df_rmout[['MONEY_VALUE','RISK_D']].describe()


# In[20]:


## Removing rest of the outliers
Mainfinal_df = Mainrisk_df_rmout[(Mainrisk_df_rmout['INHERENT_RISK'] != 622.838000) & (Mainrisk_df_rmout['TOTAL'] != 191.360000) & (Mainrisk_df_rmout['MONEY_VALUE'] != 935.030000) & (Mainrisk_df_rmout['RISK_D'] != 561.018000)]


# In[21]:


plt.boxplot(Mainfinal_df['INHERENT_RISK'])


# In[22]:


Mainfinal_df.shape


# In[23]:


## RISK_x and RISK_y are formed for Risk due to merging of two dataframes. Now OR operation is performed to built RISK column.

Mainfinal_df['RISK'] = Mainfinal_df['RISK_x'] | Mainfinal_df['RISK_y']
Mainfinal_df = Mainfinal_df.drop(['RISK_x','RISK_y'],axis=1)
Mainfinal_df.head()


# In[24]:


Mainfinal_df.describe()
# Mainfinal_df.info()


# In[25]:


Mainfinal_df.columns


# In[26]:


## DISTRIC_LOSS and DISTRICT have same values and same affect on the target, so i am dropping
## one of the two attributes (DISTRICT). It is also observed that MONEY_MARKS and SCORE_MV differ by a constant factor which is the multiplication of 10.
## SCORE_MV is 10 times of MONEY_MARKS, so MONEY_MARKS will be dropped.

Mainfinal_df = Mainfinal_df.drop(['MONEY_MARKS','DISTRICT'],axis=1)


# ## Visualizations of variables distribution

# In[27]:


## Plotting the SECTOR_SCORE vs RISK

sns.countplot(x='SECTOR_SCORE',data=Mainfinal_df[['SECTOR_SCORE','RISK']],
              hue="RISK").set_title("Sector_score Vs Risk")
plt.xticks(rotation=45)


# In[28]:


## Plotting RISK vs LOCATION

fig = plt.figure(figsize=(20,20))
sns.countplot(x='LOCATION_ID',data=Mainfinal_df[['LOCATION_ID','RISK']],
              hue="RISK").set_title("LOCATION_ID Vs RISK")
plt.xticks(rotation=45)


# In[29]:


## Plot for History vs Risk
sns.countplot(x='HISTORY',data=Mainfinal_df[['HISTORY','RISK']],
              hue="RISK").set_title("HISTORY Vs RISK")
plt.xticks(rotation=45)

## For zero history, the risk is less i.e., risk is zero.


# In[30]:


## Plot for District-LOSS vs Risk
sns.countplot(x='DISTRICT_LOSS',data=Mainfinal_df[['DISTRICT_LOSS','RISK']],
              hue="RISK").set_title("DISTRICT_LOSS Vs RISK")
plt.xticks(rotation=45)

## District-loss = 2 has less risk as risk=0.


# In[31]:


## Plot for numbers and risk

sns.countplot(x='NUMBERS',data=Mainfinal_df[['NUMBERS','RISK']],
              hue="RISK").set_title("NUMBERS Vs RISK")
plt.xticks(rotation=45)

## NUMBERS is Numbers of Transcations. Risk zero is concentrated in transactions 5.


# In[32]:


## Plot for Risk percentage
sns.countplot(x='RISK',data=Mainfinal_df[['RISK']],
              hue="RISK").set_title(" NO RISK VS RISK")
plt.xticks(rotation=45)

## Risk 1 is higher than Risk 0.


# In[33]:


## Heatmap of correlation coefficient 

plt.figure(figsize = (25,25))
sns.heatmap(Mainfinal_df.corr(), square = True, linecolor = 'red', annot = True)
Mainfinal_df.shape


# In[34]:


# Create correlation matrix
corr_matrix = Mainfinal_df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[35]:


Mainfinal_df = Mainfinal_df.drop(Mainfinal_df[to_drop], axis=1)
Mainfinal_df.head()


# In[36]:


Mainfinal_df.shape


# In[ ]:


## 11 columns are dropped from above code due to high correlation (i.e., >0.95)


# ## Scaling the features and splitting the data into X and y
# #### data is splitted into scale_x_df and y_regFinal (target for regression). 
# #### Feature scaling is performed using MinMaxScaler

# In[40]:


from sklearn.preprocessing import MinMaxScaler

Mainfinal_df1 = Mainfinal_df.copy()
mm_scaler = MinMaxScaler()

y_regFinal = Mainfinal_df['AUDIT_RISK']# Regression y
y_clfFinal = Mainfinal_df['RISK'] # Classification y

scale_x_df = Mainfinal_df1.drop(["AUDIT_RISK","RISK"], axis =1)

mm_x_df = scale_x_df.copy()

num_cols = ['SECTOR_SCORE', 'LOCATION_ID', 'PARA_A', 'SCORE_A', 'PARA_B', 'SCORE_B',
       'NUMBERS', 'SCORE_B.1', 'MONEY_VALUE', 'SCORE_MV', 'DISTRICT_LOSS',
       'PROB1', 'RISK_E', 'HISTORY', 'PROB', 'SCORE', 'INHERENT_RISK',
       'CONTROL_RISK',]

mm_x_df[num_cols] = mm_scaler.fit_transform(mm_x_df[num_cols])       # MinMax scaled X

X=mm_x_df[num_cols]
y=y_regFinal


# In[41]:


X.columns
X.shape
y.shape


# ## Regression Models

# In[42]:


from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))


# ### Linear Regression

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

lreg = LinearRegression()
lreg.fit(X_trainval, y_trainval)

print('Train score: %.4f'%lreg.score(X_trainval, y_trainval))
print('Test score: %.4f'%lreg.score(X_test, y_test))

# The coefficients
#print('Coefficients: \n', lreg.coef_)
predictions = lreg.predict(X_test)
plt.scatter(y_test, predictions)

#plot 1
plt.scatter(y_test, predictions)

#plot 2: residual
plt.scatter(predictions, predictions - y_test, c = 'b')
plt.ylabel('Residuals')
plt.title('Residuals plot of test data(dark blue) and predicted-Test Data(Orange)')
plt.show()


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

mse=metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

sns.distplot((y_test-predictions), bins=500)
coeffecients = pd.DataFrame(lreg.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:


## Linear Regression shows training score of 0.8334 and test score of 0.6136, showing poor scores.
## error values are higher.


# ### K-NN Regression

# In[44]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

knn = KNeighborsRegressor()

#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])
k_range = [1,3,5,7,9,12,15,17,20]          
weights_range = ['uniform','distance'] 
param_grid = dict(n_neighbors=k_range, weights = weights_range)
best_score = 0

#grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[45]:


train_score_array = []
test_score_array = []

knn_reg = KNeighborsRegressor(17)
knn_reg.fit(X_trainval, y_trainval)
train_score_array.append(knn_reg.score(X_trainval, y_trainval))
test_score_array.append(knn_reg.score(X_test, y_test))
print(train_score_array)
print(test_score_array)


# In[46]:


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

print('\ntrain score: ', knn_reg.score(X_trainval, y_trainval))
print('test score: ', knn_reg.score(X_test, y_test) )


# In[ ]:


## Train/Test scores of KNN regressors are very low, and the errors are high.


# ## RIDGE

# In[47]:


from  sklearn.linear_model import Ridge

x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_trainval,y_trainval)
    train_score_list.append(ridge.score(X_trainval,y_trainval))
    test_score_list.append(ridge.score(X_test, y_test))
    
print(train_score_list)
print(test_score_list)


# In[48]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import Lasso

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


# In[49]:


lasso = Lasso(alpha = 0.001)
lasso.fit(X_trainval,y_trainval)
print('Train score: {:.4f}'.format(lasso.score(X_trainval,y_trainval)))
print('Test score: {:.4f}'.format(lasso.score(X_test, y_test)))


# In[50]:


train_score_list = []
test_score_list = []

lasso = Lasso(alpha=0.001)
lasso.fit(X,y)

test_score_list.append(lasso.score(X, y))


# In[51]:


print(test_score_list)

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


# In[ ]:


## Train/Test score are low indicating Ridge Regression is not good model for the data.


# ## POLYNOMIAL

# In[52]:


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
    trainScore = lreg.score(X_train_poly, y_trainval)
    testScore = lreg.score(X_test_poly, y_test)
    
print(train_score_list)
print(test_score_list)


# In[53]:


x_axis = range(1,3)
plt.plot(x_axis, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_axis, test_score_list, c = 'b', label = 'Test Score')
plt.xlabel('degree')
plt.ylabel('accuracy')
plt.legend()


# ### DecisionTree Regression

# In[54]:


from sklearn.tree import DecisionTreeRegressor
import os
import mglearn


DT_r = DecisionTreeRegressor()
DT_r.fit(X_trainval,y_trainval)

DT_tr_pred = DT_r.predict(X_trainval)
DT_test_pred = DT_r.predict(X_test)
lreg = LinearRegression().fit(X_trainval, y_trainval)

pred_lr = lreg.predict(X_trainval)
pred_test =lreg.predict(X_test)
print('Train Score:',DT_r.score(X_trainval,y_trainval))      
print('Train Score:',DT_r.score(X_test, y_test))


# In[55]:


from sklearn import  metrics
pred_lr = lreg.predict(X_trainval)
pred_test =lreg.predict(X_test)
pred_lr_mse = metrics.mean_squared_error(y_trainval,pred_lr)
pred_lr_rmse = np.sqrt(pred_lr_mse)
pred_test_mse = metrics.mean_squared_error(y_test, pred_test)
pred_test_rmse = np.sqrt(pred_test_mse)

print('train mse: ', pred_lr_mse)
print('train rmse: ', pred_lr_rmse)

print('test mse: ', pred_test_mse)
print('test rmse: ', pred_test_rmse)


# In[ ]:


### Train score is high (0.99) but Test score is low (0.73). This indicates overfitting problem in DecisionTree Regressor.


# ## SVM

# In[56]:


from sklearn.model_selection import GridSearchCV
#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])

from sklearn import svm
from sklearn.svm import SVR
import numpy as np

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
 
svm_r = svm.SVR()
grid_search = GridSearchCV(svm_r, param_grid, cv=5, return_train_score=True,)
    
#grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search = GridSearchCV(svm_r, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[57]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
svm_r = svm.SVR(kernel='linear', C = 100)
svm_r.fit(X_trainval, y_trainval)

svmr_tr_pred = svm_r.predict(X_trainval)
svmr_test_pred = svm_r.predict(X_test)

print('Train Score for Linear Kernel:',svm_r.score(X_trainval,y_trainval))      
print('Train Score for Linear Kernel:',svm_r.score(X_test, y_test))      

svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)

print('\ntrain mse: ', svm_tr_mse)
print('train rmse: ', svm_tr_rmse)

print('test mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)


# ## SVM- Poly Kernel

# In[58]:


svm_r = svm.SVR(kernel='poly', C = 100, degree=3)
svm_r.fit(X_trainval, y_trainval)

svmr_tr_pred = svm_r.predict(X_trainval)
svmr_test_pred = svm_r.predict(X_test)

svm_tr_mse = metrics.mean_squared_error(y_trainval, svmr_tr_pred)
svm_tr_rmse = np.sqrt(svm_tr_mse)
svm_test_mse = metrics.mean_squared_error(y_test, svmr_test_pred)
svm_test_rmse = np.sqrt(svm_test_mse)

print('Train Score for Poly Kernel:',svm_r.score(X_trainval,y_trainval))      
print('Train Score for Poly Kernel:',svm_r.score(X_test, y_test)) 

print('\ntrain mse: ', svm_tr_mse)
print('train rmse: ', svm_tr_rmse)

print('test mse: ', svm_test_mse)
print('test rmse: ', svm_test_rmse)


# ## CLASSIFICATION

# In[66]:


## lets use 'RISK' as y variable for classification
y = y_clfFinal
X = mm_x_df[num_cols]


# ### K-NN CLASSIFICATION

# In[67]:


# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)

# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0


# ### KNN Classification

# In[69]:


from sklearn.neighbors import KNeighborsClassifier

train_score_array = []
test_score_array = []

for k in range(1,20):
    knn = KNeighborsClassifier(k)
    knn.fit(X_trainval, y_trainval)
    train_score_array.append(knn.score(X_trainval, y_trainval))
    test_score_array.append(knn.score(X_test, y_test))


# In[70]:


x_axis = range(1,20)

plt.plot(x_axis, train_score_array, label = 'Train Score', c = 'g')
plt.plot(x_axis, test_score_array, label = 'Test Score', c='b')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()


# #####  Now, lets use GridSearchCV with the knn classification to derive the best parameters i.e., n_neighbors and weights. Results show best n_neighbors is 1, which leaves the model with an accuracy and precision of 1 stating that the model is a perfect fit without any misclassifications.

# In[71]:


from sklearn.model_selection import cross_val_score , GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

#param_grid = dict(k_range' : [1,3,5,7,9,12,15,17,20])
k_range = [1,3,5,7,9,12,15,17,20]          
weights_range = ['uniform','distance'] 
param_grid = dict(n_neighbors=k_range, weights = weights_range)


#grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search = GridSearchCV(knn, param_grid, cv=10, return_train_score=True)
grid_search.fit(X_trainval, y_trainval)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[75]:


from sklearn.neighbors import KNeighborsClassifier

train_score_array = []
test_score_array = []


knn = KNeighborsClassifier(1) 
knn.fit(X_train, y_train)
train_score_array.append(knn.score(X_trainval, y_trainval))
test_score_array.append(knn.score(X_test, y_test))


# In[73]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score


# In[76]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

knn_c_bst_clf = KNeighborsClassifier(n_neighbors=1)

knn_c_bst_clf.fit(X_trainval,y_trainval)

knnc_tr_pred = knn_c_bst_clf.predict(X_trainval)
knnc_test_pred = knn_c_bst_clf.predict(X_test)
print(knnc_tr_pred[4])

print("Train data")

print("Accuracy score: ", accuracy_score(y_trainval, knnc_tr_pred))
print("f1 score: ", f1_score(y_trainval, knnc_tr_pred))
print("recall score: ", recall_score(y_trainval, knnc_tr_pred))
print("precision: ", precision_score(y_trainval, knnc_tr_pred))
print("   ")

print("Test data")
print("Accuracy score: ", accuracy_score(y_test, knnc_test_pred))
print("f1 score: ", f1_score(y_test, knnc_test_pred))
print("recall score: ", recall_score(y_test, knnc_test_pred))
print("precision: ", precision_score(y_test, knnc_test_pred))

confusion = confusion_matrix(y_test, knnc_test_pred)
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, knnc_test_pred))


# In[79]:


## Evaluation metrics

print(pd.crosstab(y_trainval, knnc_tr_pred))
print(pd.crosstab(y_test, knnc_test_pred))
from sklearn.metrics import classification_report
report = classification_report(y_test, knnc_test_pred)
print(report)


# In[80]:


## Calculating the roc and plotting roc curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# fit a model
knn_c_bst_clf.fit(X_trainval,y_trainval)
# predict probabilities
probs = knn_c_bst_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
print( thresholds )
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()


# In[ ]:


## all the scores of KNN shows a perfect match, and the roc curve is flat on the top of the diagram. KNN scores are not realistic.


# In[81]:


## Precision-Recall curve

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

# predict probabilities
probs = knn_c_bst_clf.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# predict class values
yhat = knn_c_bst_clf.predict(X_test)

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# calculate F1 score
f1 = f1_score(y_test, yhat)

# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()


# ###  LOGISTIC REGRESSION

# In[82]:


from sklearn.linear_model import LogisticRegression

c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_score_l1 = []
train_score_l2 = []
test_score_l1 = []
test_score_l2 = []

for c in c_range:
    log_l1 = LogisticRegression(penalty = 'l1', C = c)
    log_l2 = LogisticRegression(penalty = 'l2', C = c)
    log_l1.fit(X_trainval, y_trainval)
    log_l2.fit(X_trainval, y_trainval)
    train_score_l1.append(log_l1.score(X_trainval, y_trainval))
    train_score_l2.append(log_l2.score(X_trainval, y_trainval))
    test_score_l1.append(log_l1.score(X_test, y_test))
    test_score_l2.append(log_l2.score(X_test, y_test))
    
    


# In[83]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(c_range, train_score_l1, label = 'Train score, penalty = l1')
plt.plot(c_range, test_score_l1, label = 'Test score, penalty = l1')
plt.plot(c_range, train_score_l2, label = 'Train score, penalty = l2')
plt.plot(c_range, test_score_l2, label = 'Test score, penalty = l2')
plt.legend()
plt.xlabel('Regularization parameter: C')
plt.ylabel('Accuracy')
plt.xscale('log')


# In[84]:


### Lets do grid search

c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
penalty_mod = ['l1','l2']

log_reg = LogisticRegression()

#create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(penalty=penalty_mod,C=c_range)
print(param_grid)

#instantiation of the grid
log_reg_grid = GridSearchCV(log_reg,param_grid, cv=10, scoring='accuracy')

# fitting the grid
log_reg_grid.fit(X, y)


# In[85]:


log_reg_grid.best_score_
log_reg_grid.best_params_


# In[86]:


scores = cross_val_score(log_reg, X, y,cv=10)
# input arguments followed by X and Y
print("Cross-validation scores: {}".format(scores))


# In[87]:


log_reg = LogisticRegression(penalty = 'l1', C = 10)
log_reg.fit(X_trainval, y_trainval)

print(log_reg.score(X_trainval, y_trainval))
print(log_reg.score(X_test, y_test))


logreg_tr_pred = log_reg.predict(X_trainval)
logreg_test_pred = log_reg.predict(X_test)


# In[88]:


### Evaluation metrics

print(pd.crosstab(y_trainval, logreg_tr_pred))
print(log_reg.score(X_trainval, y_trainval))
print(pd.crosstab(y_test, logreg_test_pred))
print(log_reg.score(X_test, y_test))
report = classification_report(y_test, logreg_test_pred)
print(report)


# In[90]:


from sklearn.metrics import accuracy_score

print("Accuracy score: ", accuracy_score(y_trainval, logreg_tr_pred))
print("f1 score: ", f1_score(y_trainval, logreg_tr_pred))
print("recall score: ", recall_score(y_trainval, logreg_tr_pred))
print("precision: ", precision_score(y_trainval, logreg_tr_pred))
print("   ")
print("Test data")
print("Accuracy score: ", accuracy_score(y_test, logreg_test_pred))
print("f1 score: ", f1_score(y_test, logreg_test_pred))
print("recall score: ", recall_score(y_test, logreg_test_pred))
print("precision: ", precision_score(y_test, logreg_test_pred))

## Calculating and plotting ROC curve

# predict probabilities
probs = log_reg.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
print( thresholds )
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()


## Precision and Recall curve
from sklearn.metrics import auc
# predict probabilities
probs = log_reg.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# predict class values
y_prd_class_val = log_reg.predict(X_test)

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# calculate F1 score
f1 = f1_score(y_test, y_prd_class_val)

# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(y_test, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()


# ### LINEAR SVM

# In[91]:


from sklearn.svm import LinearSVC

c_range= [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = dict(C=c_range)
print("Parameter grid:\n{}".format(param_grid))

clf = LinearSVC()
linearsvc_grid_search = GridSearchCV(estimator=clf, param_grid = dict(C=c_range)   ,n_jobs=-1)
linearsvc_grid_search.fit(X, y)

linearsvc_grid_search.best_score_
linearsvc_grid_search.best_params_


# In[92]:


clf_best = LinearSVC(C=1)

clf_best.fit(X_trainval, y_trainval)

clf_tr_pred = clf_best.predict(X_trainval)
clf_test_pred = clf_best.predict(X_test)


# In[93]:


print("Train data")
print("Accuracy score: ", accuracy_score(y_trainval, clf_tr_pred))
print("f1 score: ", f1_score(y_trainval, clf_tr_pred))
print("recall score: ", recall_score(y_trainval, clf_tr_pred))
print("precision: ", precision_score(y_trainval, clf_tr_pred))
print("   ")
print("Test data")
print("Accuracy score: ", accuracy_score(y_test, clf_test_pred))
print("f1 score: ", f1_score(y_test, clf_test_pred))
print("recall score: ", recall_score(y_test, clf_test_pred))
print("precision: ", precision_score(y_test, clf_test_pred))


# In[94]:


### Evaluation metrics

print(pd.crosstab(y_trainval, clf_tr_pred))
print(pd.crosstab(y_test, clf_test_pred))
report = classification_report(y_test, clf_test_pred)
print(report)


# In[ ]:


### Linear SVC shows train/test scores close to 1, showing a perfect match.


# ### SVC Linear Kernel

# In[96]:


from sklearn import svm
from sklearn.svm import SVC

c_range= [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = dict(C=c_range)
print("Parameter grid:\n{}".format(param_grid))

svc = SVC(kernel='linear')
grid_search = GridSearchCV(estimator=svc, param_grid = dict(C=c_range) ,n_jobs=-1)
grid_search.fit(X, y)


# In[97]:


grid_search.best_score_
grid_search.best_params_


# In[98]:


svc_best = SVC(C=1.0, gamma='auto',probability=True)


# In[99]:


svc_best.fit(X_trainval, y_trainval)

svc_tr_pred = svc_best.predict(X_trainval)
svc_test_pred = svc_best.predict(X_test)


# In[100]:


print("Train data")
print("Accuracy score: ", accuracy_score(y_trainval, svc_tr_pred))
print("f1 score: ", f1_score(y_trainval, svc_tr_pred))
print("recall score: ", recall_score(y_trainval, svc_tr_pred))
print("precision: ", precision_score(y_trainval, svc_tr_pred))
print("   ")
print("Test data")
print("Accuracy score: ", accuracy_score(y_test, svc_test_pred))
print("f1 score: ", f1_score(y_test, svc_test_pred))
print("recall score: ", recall_score(y_test, svc_test_pred))
print("precision: ", precision_score(y_test, svc_test_pred))


# In[101]:


### Evaluation metrics

print(pd.crosstab(y_trainval, svc_tr_pred))

print(pd.crosstab(y_test, svc_test_pred))

report = classification_report(y_test, svc_test_pred)
print(report)


# In[102]:


# predict probabilities
probs = svc_best.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
print( thresholds )
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()


# In[ ]:


### SVC with linear kernel shows train/test score of 0.92/0.95, which is better regression models. 


# ### SVC Kernel RBF

# In[103]:


c_range= [0.001, 0.01, 0.1, 1, 10, 100]
gamma_range=[0.001, 0.05,0.07,0.03,0.01,0.5,0.3, 0.1, 1, 10, 100]

param_grid = dict(C=c_range, gamma=gamma_range)
print("Parameter grid:\n{}".format(param_grid))

svc = SVC(kernel='rbf')
grid_search = GridSearchCV(estimator=svc, param_grid = dict(C=c_range,gamma=gamma_range) ,n_jobs=-1)
grid_search.fit(X, y)


# In[104]:


grid_search.best_score_
grid_search.best_params_


# In[105]:


svc_best_rbf = SVC(kernel='rbf',C=10, gamma=0.7)


# In[106]:


svc_best_rbf.fit(X_trainval, y_trainval)

svc_rbf_tr_pred = svc_best_rbf.predict(X_trainval)
svc_rbf_test_pred = svc_best_rbf.predict(X_test)

print("Train data")
print("Accuracy score: ", accuracy_score(y_trainval, svc_rbf_tr_pred))
print("f1 score: ", f1_score(y_trainval, svc_rbf_tr_pred))
print("recall score: ", recall_score(y_trainval, svc_rbf_tr_pred))
print("precision: ", precision_score(y_trainval, svc_rbf_tr_pred))
print("   ")
print("Test data")
print("Accuracy score: ", accuracy_score(y_test, svc_rbf_test_pred))
print("f1 score: ", f1_score(y_test, svc_rbf_test_pred))
print("recall score: ", recall_score(y_test, svc_rbf_test_pred))
print("precision: ", precision_score(y_test, svc_rbf_test_pred))


# In[107]:


## Evaluation metrics 
print(pd.crosstab(y_trainval, svc_rbf_tr_pred))
print(pd.crosstab(y_test, svc_rbf_test_pred))

report = classification_report(y_test, svc_rbf_test_pred)
print(report)


# In[ ]:


### SVC with RBF kernel shows all perfect scores, which is not true. So, SVC with RBF kernel won't be considered as the best model.


# ### SVC Kernel Poly

# In[108]:


c_range= [0.001, 0.01, 0.1, 1, 10, 100]
degree_range=[1,2,3,4]

param_grid = dict(C=c_range, degree = degree_range)
print("Parameter grid:\n{}".format(param_grid))

grid_search.best_score_
grid_search.best_params_


# In[109]:


svc_best_poly = SVC(kernel='poly',C=10, degree=1)

svc_best_poly.fit(X_trainval, y_trainval)

svc_poly_tr_pred = svc_best_poly.predict(X_trainval)
svc_poly_test_pred = svc_best_poly.predict(X_test)

print("Train data")
print("Accuracy score: ", accuracy_score(y_trainval, svc_poly_tr_pred))
print("f1 score: ", f1_score(y_trainval, svc_poly_tr_pred))
print("recall score: ", recall_score(y_trainval, svc_poly_tr_pred))
print("precision: ", precision_score(y_trainval, svc_poly_tr_pred))
print("   ")
print("Test data")
print("Accuracy score: ", accuracy_score(y_test, svc_poly_test_pred))
print("f1 score: ", f1_score(y_test, svc_poly_test_pred))
print("recall score: ", recall_score(y_test, svc_poly_test_pred))
print("precision: ", precision_score(y_test, svc_poly_test_pred))


# In[110]:


### Evaluation metrics

print(pd.crosstab(y_trainval, svc_poly_tr_pred))
print(pd.crosstab(y_test, svc_rbf_test_pred))

report = classification_report(y_test, svc_poly_test_pred)
print(report)


# In[ ]:


## SVC with polynomial kernel shows higher train/test score and the precision, and recall score are 1.00. So SVC with polynomial kernel
## will be considered as the best model for our data.


# ### Decision Tree

# In[111]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
param_grid = dict(max_depth=[4,6,8,10])

gs_dt = GridSearchCV(DT, param_grid=param_grid, cv=10, scoring='accuracy')
gs_dt.fit(X, y)

gs_dt.best_score_
gs_dt.best_params_


# In[112]:


dt_best = DecisionTreeClassifier(max_depth=4)
dt_best.fit(X_trainval, y_trainval)

dt_tr_pred = dt_best.predict(X_trainval)
dt_test_pred = dt_best.predict(X_test)


# In[113]:


print("Train data")
print("Accuracy score: ", accuracy_score(y_trainval, dt_tr_pred))
print("f1 score: ", f1_score(y_trainval, dt_tr_pred))
print("recall score: ", recall_score(y_trainval, dt_tr_pred))
print("precision: ", precision_score(y_trainval, dt_tr_pred))
print("   ")
print("Test data")
print("Accuracy score: ", accuracy_score(y_test, dt_test_pred))
print("f1 score: ", f1_score(y_test, dt_test_pred))
print("recall score: ", recall_score(y_test, dt_test_pred))
print("precision: ", precision_score(y_test, dt_test_pred))


# In[115]:


fea_imp = dt_best.feature_importances_
columns = X_trainval.columns
feat_cols = pd.DataFrame({'name_col':columns,'feat_imp':fea_imp})


# In[116]:


features=columns
importances = dt_best.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')


# In[117]:


### Evaluation metrics

print(pd.crosstab(y_trainval, dt_tr_pred))
print(pd.crosstab(y_test, dt_test_pred))

report = classification_report(y_test, dt_test_pred)
print(report)


# In[119]:


### ROC curve

# predict probabilities
probs = dt_best.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
print( thresholds )
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()

# predict probabilities
probs = dt_best.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]

# predict class values
y_prd_class_val = dt_best.predict(X_test)

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# calculate F1 score
f1 = f1_score(y_test, y_prd_class_val)

ap = average_precision_score(y_test, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()


# In[ ]:


## Decision tree's scores are perfect, far from truth. The importance figure shows only
## score is importance in the model. We won't consider the decision tree as the best model.


# In[ ]:




