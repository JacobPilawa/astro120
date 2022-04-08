#!/usr/bin/env python
# coding: utf-8

# # Loops and Parallelism

# ## QUICK TIP!

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import time


# ## Background

# * Loops, in general, are slow and hard to read. 
# 
# * There are packages (numpy) that make computations extremely fast and efficient. Instead of performing operations on individual elements of say a vector, we instead perform operations on the *entire object* at once. 

# ## Base Python Example: Multiplying Vectors (element-wise)

# In[3]:


def multiply_arrays(arr1,arr2):
    # initialize output array
    out = []
    
    # start stop watch
    s = time.time()
    
    # multiply each element with FOR LOOP
    for value1, value2 in zip(arr1, arr2):
        out.append(value1*value2)
    
    # end stop watch
    e = time.time()
    
    # get time difference
    dt = e-s
    
    return dt, out

# call function on 0 to 10,000,000
arr1 = [i for i in range(10000000)]
arr2 = [i for i in range(10000000)]
dt_base, out = multiply_arrays(arr1,arr2)

print(dt_base,' seconds')
print(out[0:100])


# ## Numpy: Parallel Computations

# In[4]:


import numpy as np

def multiply(arr1,arr2):
    '''
    MUST INPUT NUMPY ARRAYS
    '''
    
    s = time.time()
    out = arr1 * arr2
    e = time.time()
    
    dt = e-s
    
    return dt, out

arr1 = np.arange(0,10000000)
arr2 = np.arange(0,10000000)

dt_numpy, out = multiply(arr1,arr2)

print(dt_numpy,' seconds')
print(out[0:100])


# In[5]:


print(f'Numpy is {dt_base/dt_numpy} times as fast!')


#  

#  

#  

#  

#  

#  

#  

# ## If you find yourself using a loop, numpy probably has a function!

#  

#  

#  

#  

#  

#  

#  

# ## A more practical example...applying a mask to data!

# Imagine you have a spectrum (flux vs. wavelength) from a source, and you know some parts of the data are unreliable (for whatever reason) and you want to set those values to 0.

# In[6]:


from astropy.io import fits 
import matplotlib.pyplot as plt

def get_spectrum_and_wavelengths(file):
    '''
    this code chunk reads data from a fits file
    '''
    with fits.open(file) as f:
        lam = 10**(f[0].header['COEFF0'] + f[0].header['COEFF1']*np.arange(0,3864))
        spec = f[0].data[0]
    
    return lam, spec

lam, spec = get_spectrum_and_wavelengths('spSpec-51788-0401-161.fit')

# plot 
plt.figure(figsize=(8,4),dpi=150)
plt.fill_between(x=[6000,7000],y1=-10,y2=300,color='red',alpha=0.25,label='bad region')
plt.plot(lam,spec,lw=0.5)
plt.xlabel('$\\lambda$ [angstroms]',fontsize=16)
plt.ylabel('$F_\\lambda$ [10$^{-17}$ erg/s/cm$^2$/A]',fontsize=16)
plt.ylim(-10,300)
plt.legend()
plt.show()


# In[7]:


# instead of iterating through wavelengths, checking with some if/else statements, and reassigning,
# use np.where()
is_bad = np.where((lam > 6000) & (lam < 7000))

# define the mask
mask = np.ones(len(spec))
mask[is_bad] = False

# test speeds
dt_slow, good_spectrum_slow = multiply_arrays(spec,mask)
dt_fast, good_spectrum_fast = multiply(spec,mask)


# In[8]:


print(dt_slow)
print(dt_fast)
print(dt_slow/dt_fast)


# In[9]:


# here 
plt.figure(figsize=(8,4),dpi=100)
plt.plot(lam,spec,lw=0.5,label='unmasked spectrum')
plt.plot(lam,good_spectrum_fast,lw=0.5,label='masked spectrum')
plt.xlabel('$\\lambda$ [angstroms]',fontsize=16)
plt.ylabel('$F_\\lambda$ [10$^{-17}$ erg/s/cm$^2$/A]',fontsize=16)
plt.ylim(-10,300)
plt.legend()
plt.show()


# In[ ]:




