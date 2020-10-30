#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[51]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\skyserver.csv')


# In[52]:


df


# In[53]:


df.drop(['objid', 'specobjid'], axis=1,inplace=True)


# In[54]:


df


# In[55]:


df.describe()


# In[56]:


df.info()


# In[57]:


sns.heatmap(df.isnull())


# # EDA

# In[73]:


df['class'].unique()


# In[59]:


sns.countplot(df['class'],palette='coolwarm',data=df)


# In[60]:


plt.figure(figsize=(12,8))
sns.countplot(x='camcol',hue='class',data=df)


# In[75]:


plt.figure(figsize=(12,8))
sns.countplot(x='u',hue='class',data=df)


# In[61]:


correlation = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(correlation,cmap="BuPu", annot=True,)


# In[62]:


df['class']=df['class'].replace(['STAR'],0)
df['class']=df['class'].replace(['GALAXY'],1)
df['class']=df['class'].replace(['QSO'],2)


# In[63]:


df['class'].unique()


# In[64]:


df.head()


# In[65]:


X = df.drop(columns=['class'], axis=1)
y = df['class']


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[70]:


SV=SVC()
SV.fit(X_train,y_train)
SV.score(X_train,y_train)
predsv=SV.predict(X_test)
print(accuracy_score(y_test,predsv))
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# In[71]:


DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
DT.score(X_train,y_train)
preddt=DT.predict(X_test)
print(accuracy_score(y_test,preddt))
print(confusion_matrix(y_test,preddt))
print(classification_report(y_test,preddt))


# # DECISION TREE CLASSIFIER SHOWS THE HIGHEST ACCURACY OF 98.7 PERCENT

# In[ ]:




