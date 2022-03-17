#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# data
df = pd.read_csv('~/Desktop/STAT_418/stat418_w22/hw2/q2/movies.csv').reset_index(drop = True)

df = df.rename(columns = {"title": "movie_title",
                          "year":"release_year",
                          "genre":"genre",
                          "rating":"average_rating",
                          "numvotes":"number_of_votes",
                          "plot" : "plot_description"
                           })

df.head()


# In[8]:


# For each release year: 
    # Count of movies
    # average rating
    # standard error
    # confidence interval
agg = df.groupby('release_year').agg({'movie_title':'count', 'average_rating':'mean'}).reset_index()
sigma = 1 # assumed in instructions
agg.loc[:,'standard_error'] = sigma / np.sqrt(agg['movie_title'])
agg.loc[:,'conf_int'] = 1.96 * agg['standard_error']


# In[17]:


# config
fig, ax = plt.subplots(figsize=(12,5))
# plotting
ax.bar(x=agg['release_year'], height=agg['movie_title'])
ax.scatter(x=agg['release_year'], y=agg['average_rating'])
ax.errorbar(x=agg['release_year'], y=agg['average_rating'], yerr=[agg['conf_int'], agg['conf_int']], linestyle = 'None')
plt.grid(True, which='both')
plt.title("Movies Plot")
# save ~ export
plt.savefig("q4_plot_movie.png")
plt.show()


# In[ ]:




