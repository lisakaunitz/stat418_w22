#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[9]:


df = pd.read_csv('~/Desktop/STAT_418/stat418_w22/hw1/q4/tvsales.csv')

# basic df exploration
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)


# In[47]:


# 4.1 Which store had the highest mean sale in 2017?

df['Date'] = pd.to_datetime(df['Date'])
dates2017 = (df['Date'] >= '2017-01-01') & (df['Date'] <= '2017-12-31')
df2017 = df.loc[dates2017]
df2017.shape
df2017.describe().loc[['mean']].max()

# *Answer:* Store S2 had the highest mean sale in 2017 with 75.56.


# In[49]:


# 4.2 Which day showed the highest variance in sales across different stores?

df.var(axis=1).values.argmax()
df.loc[310,:]

# *Answer:* November 7th, 2017 showed the highest variance in sales.


# In[59]:


# 4.3 Which year showed the highest median sale for the store S5? 

df.groupby(df['Date'].dt.year)['S5'].agg(['median'])

# *Answer:* The year 2019 showed the highest median sale for the store S5. 


# In[80]:


# 4.4 Which store recorded the highest number of sales for the largest number of days? 

columns_to_show = df[["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"]]
columns_to_show.idxmax(axis=1).mode()

# *Answer:* Store S2 recorded the hightest number of sales for the largest number of days.


# In[92]:


# 4.5 Which store ranks 5th in the cumulative number of units sold over the 3-year interval?
df.sum().sort_values(ascending=False).head(5)

# *Answer:* Store S7 ranks 5th in the cumulative number of units sold over the 3-year interval.


# In[94]:


# 4.6 Your program should create a file named repaired.csv in the directory hw1/q4 which contains the same data as TV-Sales.csv, but with “N/A” values replaced with the median sale of that store, over the entire 3-year interval. Retain the header row found in TV-Sales.csv.

newdf = df.fillna(df.median())
newdf.to_csv("newdf.csv")


# In[ ]:




