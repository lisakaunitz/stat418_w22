#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Problem Set 2: hw2, q2
import pandas as pd
import sqlite3
from scipy import stats


# In[ ]:


df = pd.read_csv('~/Desktop/movies.csv').reset_index(drop=True)


# In[ ]:


# basic df exploration 
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)


# In[ ]:


df = df.rename(columns = {"title": "movie_title",
                          "year":"release_year",
                          "genre":"genre",
                          "rating":"average_rating",
                          "numvotes":"number_of_votes",
                          "plot" : "plot_description"
                           })
df.head()


# In[ ]:


# Use pandas to read in movies.csv and then populate your database using sqlite3 python library. 
conn = sqlite3.connect('movieratings.db')
c = conn.cursor()

sqlite_query =  "INSERT INTO movies                  (movie_title, release_year, plot_description, genre, average_rating, number_of_votes)                   VALUES (?, ?, ?, ?, ?, ?);"
df_values = df.get_values()

c.executemany(sqlite_insert, df_values)
conn.commit()
conn.close()

df_new = pd.read_sql_query("SELECT * from movies", conn)
df_new.head(5)


# In[ ]:




