#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as sns
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor


# In[3]:


df = pd.read_csv("indian places.csv") 


# In[4]:


df


# In[5]:


col = ['Type','Google review rating','Number of google review in lakhs','Entrance Fee in INR']
df = df[col]
df


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df = df[df['Entrance Fee in INR']!=0]


# In[10]:


df


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df['Type'].unique()


# In[14]:


df['Type'].value_counts()


# In[15]:


df = pd.get_dummies(df,dtype='int')


# In[16]:


X = df.drop(columns = ['Entrance Fee in INR'])
Y = df['Entrance Fee in INR']


# In[18]:


X


# In[19]:


Y


# In[20]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)


# In[21]:


X_train.shape


# In[22]:


Y_train.shape


# In[23]:


X_test.shape


# In[24]:


Y_test.shape


# In[25]:


model = LinearRegression()
model.fit(X_train,Y_train)


# In[26]:


model.predict(X_test)


# In[27]:


model.score(X_train,Y_train)


# In[28]:


model.score(X_test,Y_test)


# In[29]:


model_las = Lasso(alpha = 0.1,max_iter = 1,tol = 0.001)
model_las.fit(X_train,Y_train)


# In[30]:


model_las.predict(X_test)


# In[31]:


model_las.score(X_train,Y_train)


# In[32]:


model_las.score(X_test,Y_test)


# In[33]:


model_rid = Ridge(alpha = 0.1,max_iter = 1,tol = 0.001)
model_rid.fit(X_train,Y_train)


# In[34]:


model_rid.score(X_train,Y_train)


# In[35]:


model_rid.score(X_test,Y_test)


# In[36]:


model_tree = DecisionTreeRegressor()
model_tree.fit(X_train,Y_train)


# In[37]:


model.predict(X_test)


# In[38]:


model.score(X_train,Y_train)


# In[39]:


model.score(X_test,Y_test)


# In[ ]:




