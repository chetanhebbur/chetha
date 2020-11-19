#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
from PIL import Image


# In[16]:


df1=pd.read_csv(r'C:\Users\Guess\Desktop\python project\Melbourne_housing_FULL.csv')


# # the dataxset contain 22 column, for the sake of univarint let us use one column of distance 

# In[17]:


df1=df1['Distance']


# In[18]:


df1


# In[19]:


len(df1)


# In[21]:


df1.isnull().sum()=#missing value


# In[22]:


df1=df1.dropna()#drop missing value


# In[23]:


df1


# # unvivariant analysis 

# In[25]:


plt.hist(df1, bins=50)


# In[27]:


sns.distplot(df1)


# In[28]:


sns.distplot(df1, hist=False)


# In[30]:


sns.violinplot(df1)


# let us have a closeer look at he distrubution by ploting simple histrogramm with  bins

# In[35]:


plt.figure(figsize=(20,10))#make plot wider
plt.hist(df1, color='g')#plot simple histgromma
plt.axvline(df1.mean(), color='m', linewidth=1)
plt.axvline(df1.median(), color='b', linestyle='dashed', linewidth=1)
plt.axvline(df1.mode()[0],color='w',  linestyle='dashed', linewidth=1)


# In[36]:


sns.distplot(df1, hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))#cumulative distubution 


# # mulitvariant analysis

# In[91]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
from PIL import Image


# In[92]:


df2=pd.read_csv(r'C:\Users\Guess\Desktop\python project\Melbourne_housing_FULL.csv')


# In[93]:


import seaborn as sns


# In[94]:


df2.head()


# In[95]:


df2=df2.isnull().sum()


# In[96]:


df2


# In[105]:


df2=df1.dropna()


# In[108]:


sns.pairplot('df1')
plt.show()


# In[104]:


sns.scatterplot(df1['Price'], df2['Distance'])


# In[113]:


df1.corr()


# In[115]:


df_dummies=pd.get_dummies(df1,  columns=['ParkingArea'])


# In[144]:


df_encoded = pd.DataFrame(encoded, columns=['RegionID_'+str(int(i)) for i in range (encoded.shape[1])])


# In[143]:


df_encoded.head()


# In[119]:


df_dummies.columns


# In[129]:


from sklearn.preprocessing import LabelEncoder  #improt label encoder

labelencoder = LabelEncoder()

df_dummies['RegionID']=labelencoder.fit_transform(df_dummies.Regionname)


# In[130]:


df_dummies['RegionID'].head()


# In[132]:


df_dummies['RegionID'].value_counts()


# In[133]:


df_dummies['RegionID'].unique() #unique id 


# In[134]:


df_dummies['RegionID'].nunique() no of unique id


# In[136]:


from sklearn.preprocessing import OneHotEncoder
hotencoder = OneHotEncoder()
encode = hotencoder.fit_transform(df_dummies.RegionID.values.reshape(-1,1)).toarray()


# In[137]:


encode


# In[138]:


encode.shape


# In[ ]:




