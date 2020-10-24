#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


# In[3]:


df=pd.read_csv("C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\AVOCADO.csv")
df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['type'].value_counts()


# In[7]:


df['region'].unique()


# In[8]:


df.tail()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


sns.heatmap(df.isnull())


# In[12]:



df.drop('Unnamed: 0',axis=1,inplace=True)


# In[13]:


df.head()


# In[14]:


df['Date'] = pd.to_datetime(df['Date'])


# In[15]:


df.head()


# In[16]:


column_adj=['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags']

for cname in column_adj:
    df[cname]=df[cname].abs().astype('int')


# In[17]:


df.head()


# In[18]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# In[19]:


plt.figure(figsize=[26,10])
sns.countplot(x = 'year', data = df)


# In[20]:


plt.figure(figsize=(12,5))
plt.title("Distribution of average price")
sns.distplot(df["AveragePrice"], color = 'y')


# In[21]:


df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)


# In[22]:


df.head()


# In[23]:


plt.figure(figsize=(20,10))
sns.lineplot(x="Month", y="AveragePrice", hue='type', data=df)
plt.show()


# In[24]:


plt.figure(figsize=(20,10))
sns.lineplot(x="Day", y="AveragePrice", hue='type', data=df)
plt.show()


# In[25]:


plt.figure(figsize=(20,10))
sns.lineplot(x="Day", y="AveragePrice", hue='type', data=df)
plt.show()


# In[27]:


Dategroup=df.groupby('Date').mean()
plt.figure(figsize=(10,6))
byDate['AveragePrice'].plot()
plt.title('Average Price')


# In[28]:


DF=pd.get_dummies(df.drop(['region','Date'],axis=1),drop_first=True)


# In[30]:


DF.head()


# In[33]:


x=DF.iloc[:,1:14]
y=DF['AveragePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[35]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,y_train)
predict=LR.predict(X_test)


# In[37]:


from sklearn import metrics
print('Meanabserror:', metrics.mean_absolute_error(y_test, predict))
print('Meansquareerror:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))


# In[39]:


plt.scatter(x=y_test,y=predict)


# In[40]:


#WE CAN SEE THAT WE DONT HAVE A STRAIGHT LINE SO WE WILL APPLY random forest regressor


# In[47]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train,y_train)
pred=RF.predict(X_test)


# In[49]:


print('Meanabserror', metrics.mean_absolute_error(y_test, pred))
print('Meansqerror:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[51]:


sns.distplot((y_test-pred),bins=30)


# In[ ]:


#This normal distribution is the sign that the model is working well

