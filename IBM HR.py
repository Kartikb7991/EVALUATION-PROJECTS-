#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\HR.csv')


# In[30]:


df


# In[62]:


df.info()


# In[32]:


df.describe()


# In[33]:


df.columns


# In[34]:


sns.heatmap(df.isnull())


# In[35]:


sns.countplot(x='Attrition',data=df)


# In[36]:


df['Gender'].unique()


# In[37]:


sns.factorplot(x='Attrition',y='Age',kind='bar',data=df)


# In[38]:


Att_perc = df.Attrition.value_counts() / len(df)
Att_perc


# In[39]:


sns.factorplot(x='Attrition',col='Department',data=df,kind='count',)


# In[40]:



pd.crosstab(columns=[df.Attrition],index=[df.Gender],margins=True,normalize='index')


# In[41]:


pd.crosstab(columns=[df.Attrition],index=[df.JobSatisfaction],margins=True,normalize='index')


# In[42]:


plt.figure(figsize =(10, 6)) 
sns.boxplot( x ='Attrition',y ='MonthlyIncome', data = df)


# In[43]:


df.drop('StandardHours', axis = 1, inplace = True) 
df.drop('EmployeeCount', axis = 1, inplace = True) 
df.drop('EmployeeNumber', axis = 1, inplace = True) 
df.drop('Over18', axis = 1, inplace = True) 
print(df.shape)


# In[44]:


df.head()


# In[45]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
le.fit_transform(df['Attrition']) 


# In[46]:


df.head()


# In[51]:


x = df.drop(['Attrition'], axis=1)
y=df['Attrition']


# In[56]:


x.shape


# In[57]:


y.shape


# In[60]:


plt.subplots(figsize=(12,8))
sns.countplot(x='Age',hue='Attrition',data=df)


# In[63]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
y = le.fit_transform(y) 


# In[65]:


BusinessTraveldummy = pd.get_dummies(df['BusinessTravel'],  
                                    prefix ='BusinessTravel') 
Departmentdummy = pd.get_dummies(df['Department'],  
                                prefix ='Department') 
Educationfielddummy = pd.get_dummies(df['EducationField'],  
                                    prefix ='EducationField') 
Genderdummy = pd.get_dummies(df['Gender'],  
                            prefix ='Gender', drop_first = True) 
JobRoledummy = pd.get_dummies(df['JobRole'],  
                             prefix ='JobRole') 
MaritalStatusdummy = pd.get_dummies(df['MaritalStatus'],  
                                   prefix ='MaritalStatus') 
OverTimedummy = pd.get_dummies(df['OverTime'],  
                              prefix ='OverTime', drop_first = True) 
# Adding these dummy variable to input X 
x = pd.concat([x, BusinessTraveldummy, Departmentdummy,  
               Educationfielddummy, Genderdummy, JobRoledummy,  
               MaritalStatusdummy, OverTimedummy], axis = 1) 
# Removing the categorical data 
x.drop(['BusinessTravel', 'Department', 'EducationField',  
        'Gender', 'JobRole', 'MaritalStatus', 'OverTime'],  
        axis = 1, inplace = True) 
  
print(x.shape) 
print(y.shape) 


# In[89]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[90]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
lr = LogisticRegression()
lr.fit(x_train, y_train)
train_Pred = lr.predict(x_train)


# In[91]:


x.head()


# In[92]:


metrics.confusion_matrix(y_train,train_Pred)


# In[93]:


metrics.accuracy_score(y_train,train_Pred)


# In[95]:


Predtest = lr.predict(x_test)


# In[99]:


metrics.confusion_matrix(y_test,Predtest)


# In[100]:


metrics.accuracy_score(y_test,Predtest)


# In[101]:


from sklearn.metrics import classification_report
print(classification_report(y_test, Predtest))


# In[ ]:




