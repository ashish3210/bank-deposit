#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost


# In[17]:


df = pd.read_csv('gs://cloud-ml-tables-data/bank-marketing.csv') 


# In[18]:


df = df[['Age', 'Balance', 'Day', 'Duration', 'Campaign', 'PDays', 'Previous', 'Deposit']]


# In[19]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(df.drop('Deposit', axis = 1), df.Deposit)


# In[20]:


y_pred = rf_classifier.predict(df.drop('Deposit', axis = 1))


# In[ ]:





# In[ ]:




