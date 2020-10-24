#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


df=pd.read_csv("C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\HEART DSS.csv")
df


# In[20]:


df.describe()


# In[21]:


df.info()


# In[22]:


df.corr()


# In[23]:


sns.heatmap(df.corr(),annot=True)


# In[24]:


df.isnull().sum()


# In[25]:


sns.heatmap(df.isnull())


# In[26]:


df.target.unique()


# In[27]:


df.target.value_counts()


# In[28]:


sns.countplot(df.target)


# In[29]:


sns.countplot(df.sex)


# In[30]:


sns.barplot(df['sex'],df['target'])


# In[31]:


df.cp.unique()


# In[32]:


sns.barplot(df["cp"],df['target'])


# In[33]:


sns.countplot(x='target', hue='sex', data=df)


# In[34]:


plt.figure(figsize=(8,4))
sns.countplot(x='target', hue='sex', data=df)
plt.xlabel('0 - No Disease, 1 - Disease')
plt.title('Distribution of disease  among sexes')


# In[35]:


plt.figure(figsize=(10,6))
sns.countplot(x='cp', hue='target', data=df)
plt.xlabel('Type of CP')
plt.legend(['No Disease', 'Disease'])
plt.show()


# In[39]:


df.hist(figsize=(20,20))


# In[40]:



sns.barplot(df['fbs'],df['target'])


# In[41]:



df["restecg"].unique()


# In[45]:


sns.barplot(df['restecg'],df['target'])


# In[44]:



df["exang"].unique()


# In[46]:


sns.barplot(df['exang'],df['target'])


# In[47]:


df["slope"].unique()


# In[48]:


sns.barplot(df['slope'],df['target'])


# In[49]:


df["ca"].unique()


# In[50]:


sns.barplot(df['ca'],df['target'])


# In[51]:


df["thal"].unique()


# In[53]:


sns.barplot(df['thal'],df['target'])


# In[54]:



sns.distplot(df["thal"])


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

x = df.drop("target",axis=1)
y = df["target"]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# In[67]:


X_train.shape


# In[68]:


X_test.shape


# In[69]:


Y_train.shape


# In[70]:


Y_test.shape


# In[71]:


lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)


# In[75]:


accuracy = round(accuracy_score(Y_pred,Y_test)*100,2)

print("Logistic Regression gives us an accuracy score of: "+str(accuracy)+" %")


# In[ ]:




