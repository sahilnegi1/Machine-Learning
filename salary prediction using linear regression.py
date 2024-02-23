#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("C:\\Users\\sahil\\Downloads\\csv_datasets\\salary\\Salary.csv")


# In[2]:


data


# In[157]:


x=data['YearsExperience'].values
x


# In[158]:


y=data['Salary'].values


# In[159]:



import matplotlib.pyplot as plt
import numpy as np


# In[160]:


plt.plot(x,y,'+')


# In[161]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# In[162]:


x_train.shape


# In[163]:


x_test.shape


# In[164]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[166]:


lr.fit(x_train.reshape(-1,1),y_train)


# In[182]:


a=float(input("Enter the experience to predict salary = "))


# In[183]:


y_pred=lr.predict([[a]])
print(y_pred)


# In[189]:


y_pred=lr.predict(x_test.reshape(-1,1))
y_pred


# In[190]:


y_test


# In[191]:


print(lr.coef_)


# In[192]:


print(lr.intercept_)


# In[194]:


from sklearn.metrics import r2_score
acu=r2_score(y_test,y_pred)


# In[195]:


acu


# In[19]:


plt.plot(x_test,y_test,'rx')
plt.plot(x_test,y_pred,c='b')


# In[20]:


from sklearn import metrics


# In[21]:


print('mean absolute error = ',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




