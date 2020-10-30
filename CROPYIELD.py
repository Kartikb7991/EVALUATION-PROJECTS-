#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_test=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\CROPTESTING.csv')
df_train=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\CROPTRAINING.csv')


# In[3]:


df_test.shape


# In[5]:


df_train.shape


# In[7]:


df_test.columns


# In[8]:


df_train.columns


# # Handling null values

# In[9]:


df_train.isnull().sum(),df_test.isna().sum()


# In[10]:


df_train.info(),df_test.info()


# # Correlations

# In[11]:


df_train.corr(method ='pearson')


# In[12]:


df_train['Crop_Damage'].value_counts()


# In[13]:


df_train.isnull().sum()


# In[15]:


df_test.isnull().sum()


# In[18]:


df_train['Number_Weeks_Used'] = df_train['Number_Weeks_Used'].fillna(df_train['Number_Weeks_Used'].mode()[0])


# In[19]:


df_test['Number_Weeks_Used'] = df_test['Number_Weeks_Used'].fillna(df_test['Number_Weeks_Used'].mode()[0])


# In[20]:


df_train.isnull().sum()


# In[21]:


df_test.isnull().sum()


# In[54]:


sns.heatmap(df_train.corr())


# In[53]:


sns.countplot(df_train['Number_Doses_Week'],hue=df_train['Crop_Type'])


# In[52]:


sns.countplot(df_train['Pesticide_Use_Category'],hue=df_train['Crop_Type'])


# In[51]:


sns.countplot(df_train['Season'],hue=df_train['Crop_Type'])


# In[49]:


sns.countplot(df_train['Season'],hue=df_train['Soil_Type'])


# In[47]:


sns.countplot(df_train['Season'],hue=df_train['Crop_Damage'])


# In[46]:


sns.countplot(df_train['Soil_Type'],hue=df_train['Crop_Damage'])


# In[45]:


sns.countplot(df_train['Crop_Type'],hue=df_train['Crop_Damage'])


# In[22]:


df_train.drop('ID',axis=1,inplace=True)


# In[24]:


df_test.drop('ID',axis=1,inplace=True)


# In[25]:


df_test.head()


# In[28]:


X_train, Y = df_train.drop(["Crop_Damage"], axis=1).values, df_train["Crop_Damage"].values
X_test = df_test.values


# In[29]:


X_train.shape, Y.shape, X_test.shape


# In[30]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, Y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[35]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report


# # Random Forest

# In[33]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(oob_score = True, n_estimators=100)

RFC.fit(X_train,y_train)


# In[38]:


prediction = RFC.predict(X_valid) 
print(classification_report(y_valid, prediction))

mae = mean_absolute_error(prediction,y_valid) 

print("Mean Absolute Error:" , mae)


# # Logistic Regression

# In[41]:



from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train,y_train)


# In[42]:


predictions = LR.predict(X_valid) 
print(classification_report(y_valid, predictions))

mae = mean_absolute_error(predictions,y_valid) 

print("Mean Absolute Error:" , mae)


# # LOGISTIC REGRESSION SHOWS AN ACCURACY OF 84 PERCENT

# In[ ]:




