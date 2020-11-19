#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r'C:\Users\Guess\Desktop\python\grate learning\Admission_Predict.csv')


# In[3]:


data.head()


# In[4]:


data.shape#dimminsion of   data


# In[5]:


data.info()


# In[6]:


print("data:",data.mean())


# In[7]:


print(data['GRE Score'].mode())


# In[8]:


print("data:",data.median())


# In[9]:


print("data_quantile(25%)",data.quantile(q=0.25))#print the below below the 25% of data lies


# In[10]:


print("data_quantile(25%)",data.quantile(q=0.50))#print the point 50% of data lies


# In[11]:


data.describe()


# In[12]:


data["TOEFL Score"].quantile(0.75)-data["TOEFL Score"].quantile(0.25)


# In[13]:


"create boxplot for colomn=toefl score",
data.boxplot(column='TOEFL Score', return_type="axes",figsize=(8,8))
plt.text(x=0.74, y=112.00, s="3rd quantile")
plt.text(x=0.8, y=107.00, s="median")
plt.text(x=0.75, y=103.00, s="1st quantile")
plt.text(x=0.09, y=92, s="min")
plt.text(x=0.09, y=120.00, s="max")
plt.text(x=0.07, y=107.05, s="IQR", rotation=90, size=25)


# In[14]:


data.quantile(0.75)-data.quantile(0.25)#iqr


# In[15]:


print(data.max()-data.min())#range


# In[16]:


print(data.var())#var


# In[17]:


print(data.std())


# In[18]:


data.cov()


# In[19]:


import seaborn as sns


# In[20]:


sns.pairplot(data, kind="reg")
plt.show()


# In[21]:


data.drop('Serial No.', axis=1, inplace=True)


# In[22]:


data.head()


# In[63]:


fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
plt.show()


# In[23]:


data.skew()


# In[24]:


import scipy.stats as states
#conversting pandas dataframe object to numpy array and sort
h=np.asarray(data['GRE Score'])
h=sorted(h)

# use scipy states module to fit a normal distrubution with same mean and sd
fit=states.norm.pdf(h, np.mean(h),  np.std(h))

#plot both series on histogram
plt.plot(h,fit, '-',linewidth=2, label="normal distrubution with same mean and var" )
plt.hist(h, normed=True,bins=100,label='actual distrubution')
plt.legend()
plot.show()


# In[ ]:





# In[ ]:





# In[ ]:




