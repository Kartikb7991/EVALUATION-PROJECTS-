#!/usr/bin/env python
# coding: utf-8

# In[432]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[433]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\HRANALYTICS.csv')


# In[434]:


df


# In[435]:


df.shape


# In[436]:


df.columns


# In[437]:


df.info()


# In[438]:


df.describe()


# In[439]:


df['workclass'].unique()


# # Checking for null values

# In[440]:


df.isnull().sum()


# # Data cleaning

# In[441]:



df['workclass'].value_counts()


# In[442]:



print(df['marital-status'].value_counts())


# In[443]:


df['race'].value_counts()


# In[444]:


df['gender'].value_counts()


# In[445]:


df['income'].unique()


# In[446]:


df['native-country'].value_counts()


# In[447]:


df['occupation'].value_counts()


# # Initial EDA

# In[448]:


sns.countplot(df['income'],palette='coolwarm',data=df)


# In[449]:


sns.countplot(df['gender'],palette='coolwarm',hue=df['income'],data=df)


# In[450]:


sns.countplot(df['income'],palette='coolwarm',hue=df['race'],data=df)


# In[451]:


sns.countplot(df['income'],palette='coolwarm',hue=df['relationship'],data=df)


# In[452]:


df['workclass']=df['workclass'].replace('?','Private')
df['occupation']=df['occupation'].replace('?','prof-speciality')
df['native-country']=df['native-country'].replace('?','united-states')


# In[453]:


df['workclass'].value_counts()


# In[454]:


df['occupation'].unique()


# In[455]:


df['native-country'].unique()


# In[456]:


df.head()


# # FEATURE ENGINEERING

# In[457]:


#education


# In[458]:


df['education'].value_counts()


# In[459]:


df['education']=df['education'].replace(['Preschool','1st-4th','5th-6th','9th','12th','7th-8th','10th','11th',],'school')
df['education']=df['education'].replace('HS-grad','Highschool')
df['education']=df['education'].replace(['Assoc-voc','Assoc-acdm','Prof-school','Some-college'],'higher ed')


# In[460]:


df['education'].value_counts()


# In[461]:


#Marital status


# In[462]:


df['marital-status'].unique()


# In[463]:


df['marital-status']=df['marital-status'].replace(['Married-civ-spouse','Married-AF-spouse'],'Married')
df['marital-status']=df['marital-status'].replace(['Separated','Divorced','Widowed','Married-spouse-absent'],'others')
df['marital-status']=df['marital-status'].replace('Never-married','Not Married')


# In[464]:


df['marital-status'].unique()


# In[465]:


df['income'].unique()


# In[466]:


df['income']=df['income'].replace(['<=50K'],0)
df['income']=df['income'].replace(['>50K'],1)


# In[467]:


df['income'].unique()


# # EDA

# In[468]:


df.corr()


# In[469]:


sns.heatmap(df.corr(),annot=True)


# In[470]:


df.hist(figsize=(12,10),layout=(3,3),sharex=False)


# In[471]:


df.plot(kind='box',figsize=(12,12),layout=(3,3),sharex=False,subplots=True);


# In[472]:


sns.countplot(df['education'],palette='coolwarm',hue=df['income'],data=df)


# In[473]:


df.columns


# In[474]:


sns.countplot(df['relationship'],palette='coolwarm',hue=df['income'],data=df)


# In[475]:


sns.countplot(df['marital-status'],palette='coolwarm',hue=df['income'],data=df)


# In[476]:


X=df.drop(['income'],axis=1)
y=df['income']


# In[477]:


from sklearn.preprocessing import StandardScaler,LabelEncoder


# In[478]:


X.shape


# In[479]:


y.shape


# In[480]:


X.head()


# In[481]:


y.head()


# In[482]:


le=LabelEncoder()


# In[483]:


df.head()


# In[484]:


from sklearn.model_selection import train_test_split
X_train,X_train,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# In[485]:


catcols = ['workclass',"education","marital-status",'relationship',"occupation","race","gender",'native-country']
for x in catcols:
    df[x] = le.fit_transform(df[x])


# In[486]:


df.head()


# In[487]:


sc=StandardScaler()


# In[488]:


X=sc.fit_transform(df.drop(['income'],axis=1))


# In[489]:


X.shape


# In[490]:


y.shape


# In[497]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# In[ ]:





# # LOGISTIC REGRESSION 

# In[498]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# # logistic regression

# In[499]:


LR=LogisticRegression()
LR.fit(X_train,y_train)
LR.score(X_train,y_train)
predlr=LR.predict(X_test)
print(accuracy_score(y_test,predlr))
print(confusion_matrix(y_test,predlr))
print(classification_report(y_test,predlr))


# # SVC

# In[500]:


SV=SVC()
SV.fit(X_train,y_train)
SV.score(X_train,y_train)
predsv=SV.predict(X_test)
print(accuracy_score(y_test,predsv))
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# # Decision Tree Classification

# In[501]:


DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
DT.score(X_train,y_train)
preddt=DT.predict(X_test)
print(accuracy_score(y_test,preddt))
print(confusion_matrix(y_test,preddt))
print(classification_report(y_test,preddt))


# # SVM shows the highest accuracy rate of 85 percent

# In[ ]:




