#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\Baseball.csv')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


sns.heatmap(df.isnull())


# In[8]:


sns.pairplot(df)


# In[9]:


sns.heatmap(df.corr(),annot=True)


# In[10]:


df['ERA']=df["ERA"].astype(int)


# In[11]:


df.info()


# In[12]:


df.head()


# In[13]:


from scipy.stats import zscore


# In[14]:


z=np.abs(zscore(df))
z


# In[15]:


threshold=3
print(np.where(z>3))


# In[16]:


DF=df[(z<3).all(axis=1)]


# In[17]:


DF


# In[18]:


print(df.shape)
print(DF.shape)


# In[19]:


sns.lineplot(x=DF['R'],y=DF['W'],data=df)


# In[20]:


sns.lineplot(x=DF['E'],y=DF['W'],data=df)


# In[21]:


plt.subplots(figsize=(12,8))
sns.heatmap(df.corr())


# In[22]:


sns.lineplot(x=DF['2B'],y=DF['W'],data=df)


# In[23]:


sns.lineplot(x=DF['SV'],y=DF['W'],data=df)


# In[24]:


sns.lineplot(x=DF['ERA'],y=DF['W'],data=df)


# In[25]:


plt.hist(x='W',bins=10,data=DF)


# In[26]:


DF.skew()


# In[27]:


DF['E']=np.sqrt(DF['E'])


# In[28]:


DF.skew()


# In[35]:


# splitting the data into dependent and independent variables

x = df.drop('W', axis = 1)
y = df.W

print(x.shape)
print(y.shape)


# In[47]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[48]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,y_train)
predict=LR.predict(X_test)


# In[49]:


from sklearn import metrics
print('Meanabserror:', metrics.mean_absolute_error(y_test, predict))
print('Meansquareerror:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))


# In[50]:


plt.scatter(x=y_test,y=predict)


# In[51]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train,y_train)
pred=RF.predict(X_test)


# In[52]:


print('Meanabserror', metrics.mean_absolute_error(y_test, pred))
print('Meansqerror:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[55]:


sns.distplot((y_test-pred),bins=5)


# In[ ]:


#The random forest regressor shows a better curve so as a result this model is suitable for the data

