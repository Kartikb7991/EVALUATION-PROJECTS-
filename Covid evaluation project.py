#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler,LabelEncoder


# In[83]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\COVIDCASE.csv')


# In[84]:


df


# In[85]:


df.info()


# In[86]:


df.describe()


# In[87]:


df.shape


# In[88]:


df.isnull().sum()


# In[89]:


sns.heatmap(df.isnull())


# In[90]:


df.columns


# In[91]:


df.isnull().sum()


# In[92]:


df['Lat'].dropna(inplace=True)
df['Hospitalization_Rate'].fillna(df['Hospitalization_Rate'].mean(),inplace=True)

df['Recovered'].fillna(df['Recovered'].mean(),inplace=True)


# In[93]:


df.isnull().sum()


# In[94]:


df.dropna(inplace=True)


# In[95]:


df.isnull().sum()


# In[96]:


sns.heatmap(df.isnull())


# In[97]:


df['Last_Update']=pd.to_datetime(df['Last_Update'])


# In[98]:


df.head()


# In[99]:


df['Last_Update']=df['Last_Update'].map(dt.datetime.toordinal)


# In[100]:


sns.lineplot(x='Confirmed',y='Recovered',data=df)


# In[101]:


plt.figure(figsize=(16,10))
sns.barplot(x='Confirmed',y='Recovered',data=df)


# In[102]:


sns.pairplot(df)


# In[103]:


df_active=df['Confirmed']-df['Deaths']


# In[104]:


top=df[df['Last_Update']==df['Last_Update'].max()]
world=top.groupby('Province_State')['Confirmed','Recovered','Deaths'].sum().reset_index()
world.head()


# In[105]:


top_confirmed=top.groupby(by='Province_State')['Confirmed'].sum().sort_values(ascending=False).head(20).reset_index()


# In[106]:


top_confirmed


# In[107]:


plt.figure(figsize=(15,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Confirmed',fontsize=30)
plt.ylabel('Province_State',fontsize=30)
plt.title('top 20 countries hving most confirmed cases',fontsize=30)
ax=sns.barplot(x=top_confirmed.Confirmed,y=top_confirmed.Province_State)


# In[108]:


top_Deaths=top.groupby(by='Province_State')['Deaths'].sum().sort_values(ascending=False).head(20).reset_index()


# In[109]:


plt.figure(figsize=(15,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Deaths',fontsize=30)
plt.ylabel('Country',fontsize=30)
plt.title('top 20 countries hving most Deaths',fontsize=30)
ax=sns.barplot(x=top_Deaths.Deaths,y=top_Deaths.Province_State)


# In[110]:


df.head()


# In[111]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[112]:


catcols = ['Province_State',"Country_Region","ISO3",]
for x in catcols:
    df[x] = le.fit_transform(df[x])


# In[113]:


df.head()


# In[114]:


X = df.drop(columns=['Deaths'], axis=1)
y = df['Deaths']


# In[115]:


X.shape


# In[116]:


y.shape


# In[117]:


from sklearn.preprocessing import StandardScaler


# In[118]:


sc=StandardScaler()


# In[119]:


X=sc.fit_transform(df.drop(columns=['Deaths'], axis=1))


# In[123]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[134]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# In[135]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[136]:


Y_Pred = reg.predict(X_test)
Y_Pred


# In[140]:


# finding the mean squared error and variance
mse = mean_squared_error(y_test, Y_Pred)
print('RMSE :', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, Y_Pred))


# In[ ]:




