#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[67]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\loan_prediction.csv')


# In[68]:


df


# In[69]:


df.describe()


# In[70]:


df.info()


# In[71]:


df.isnull().sum()


# In[72]:


sns.heatmap(df.isnull())


# In[73]:


df['Property_Area'].unique()


# In[74]:


df['Education'].unique()


# In[75]:


sns.countplot(df['Property_Area'],hue=df['Loan_Status'])


# In[76]:


df.boxplot()


# In[77]:


df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mean(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# In[78]:


df.isnull().sum()


# In[79]:


sns.heatmap(df.isnull())


# In[80]:


df.head()


# In[81]:


df['Dependents'].value_counts()


# In[82]:


sns.countplot(x='Gender',hue='Loan_Status',data=df)


# In[83]:


sns.countplot(x='Education',hue='Loan_Status',data=df)


# In[84]:


sns.countplot(x='Married',hue='Loan_Status',data=df)


# In[85]:


df['CoapplicantIncome'].hist()


# In[86]:


df['ApplicantIncome'].hist()


# In[87]:


sns.countplot(df['Property_Area'])


# In[88]:


df['Combined_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[89]:


df['Combined_Income'].hist()


# In[90]:


correlation = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(correlation,cmap="BuPu", annot=True,)


# In[91]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[92]:


catcols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
for x in catcols:
    df[x] = le.fit_transform(df[x])


# In[93]:


df.head()


# In[94]:


df.drop('Loan_ID',axis=1,inplace=True)


# In[95]:


df.head()


# In[96]:


X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']


# In[99]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[107]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# # Logistic Regression

# In[108]:


LR=LogisticRegression()
LR.fit(X_train,y_train)
LR.score(X_train,y_train)
predlr=LR.predict(X_test)
print(accuracy_score(y_test,predlr))
print(confusion_matrix(y_test,predlr))
print(classification_report(y_test,predlr))


# In[112]:


SV=SVC()
SV.fit(X_train,y_train)
SV.score(X_train,y_train)
predsv=SV.predict(X_test)
print(accuracy_score(y_test,predsv))
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# In[113]:


DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
DT.score(X_train,y_train)
preddt=DT.predict(X_test)
print(accuracy_score(y_test,preddt))
print(confusion_matrix(y_test,preddt))
print(classification_report(y_test,preddt))


# # Logistic Regression shows the highest accuracy
