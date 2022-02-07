#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[25]:


# Creating my array
myarray = np.array([[0.111,1.211,2.311,3.411,4.511], [-5.411,-6.311,-7.211,-8.111,-9.011]])
myarray


# In[26]:


# 6.1 A 2d array with rows of magnitude > 1
rows = myarray[0:2, :]
rows


# In[27]:


# 6.2 A 1d array with all elements of the matrix in a column-major way
myarray[:, 0]


# In[28]:


# 6.3 A 2d array with all the negative values replaced by zero
myarray[myarray < 0] = 0
myarray


# In[29]:


# 6.4 A 3x4 Block Matrix (2d array) with each block as the input 2d array
a = 3
b = 4
repetitions = (a,b)
print(np.tile(myarray, repetitions))


# In[30]:


# 6.5 A 2d array with all elements of the input array rounded to the nearest hundred
np.around(myarray, decimals=2)


# In[36]:


# Bonus: 6.7 Median of all entries in the array
print(np.median(myarray, axis = 0))
print(np.median(myarray, axis = 1))


# In[45]:


# Bonus: 6.11 A 2d array with non-diagonal entries set to zero(given that it is square)
sqarray = np.arange(16).reshape(4,4)[:,:4]

diag = np.einsum('ii -> i', sqarray)
nonzeros = diag.copy()

sqarray[...] = 0
diag[...] = nonzeros

sqarray

