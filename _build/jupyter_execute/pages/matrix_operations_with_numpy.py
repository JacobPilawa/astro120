#!/usr/bin/env python
# coding: utf-8

# # Matrix Operations with Numpy

# # Base Python makes it tough to deal with matrices. Numpy is designed to deal with matrices!

# In[1]:


import numpy as np


# Looking at a sample matrix.

# # A sample matrix and accessing elements

# In[2]:


matrix = np.array([[-2, 3, 1],
                   [0,  9, 2],
                   [3, -1, -1]])
print(matrix)
print(matrix.shape)


# ### Each element of the matrix is specificed by a row and column (zero-indexed). We can access each element with **slicing**, which looks like this.

# In[3]:


row    = 1
column = 0

print(matrix[row,column])


# ### I can get entire rows or entire columns with the colon operator, which looks like this:

# In[4]:


print(matrix[:,column])


# In[5]:


print(matrix[0:2,column]) # note that the slicing is NOT inclusive!


# # Operations

# ### Matrix Multiplication (np.matmul(), or the @ operator)

# In[6]:


x = np.array([[1,0,0],
              [0,0,0],
              [0,0,0]])

y = np.random.rand(3,3)
print(y)


# In[7]:


print(y @ x)
print(' ')
print(np.matmul(y,x))


# ### Inverses

# In[8]:


array = np.random.rand(3,3)
array_inverse = np.linalg.inv(array)


# In[9]:


print(array)
print('')
print(array_inverse)
print('')
print(array @ array_inverse)


# ### Tranposing

# In[10]:


x = np.array([[1,2,3],
             [4,5,6]])

print(x)
print('')
print(x.shape)
print('')
x_transpose = x.T # or np.tranpose()
print(x_transpose)
print('')
print(x_transpose.shape)


# In[11]:


x @ x_transpose


# In[12]:


x @ x


# Other useful functions that I use often:
# 
# * np.zeros()
# * np.ones()
# * np.hstack() -- stacks arrays columnwise
# * np.vstack() -- stack arrays rowwise
# * np.save(), np.load() -- saves and loads .npy files 
# * np.genfromtxt(), np.loadtxt(), np.savetxt()
# * np.random.uniform()
# * np.random.normal()
# * 

# In[ ]:




