#!/usr/bin/env python
# coding: utf-8

# # Reading FITS Files

# In[1]:


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[2]:


plt.rcParams['figure.dpi']=150


# # Reading a FITS File

# In[3]:


file_name = 'd165_os_bs_ff_bp_crj.fits'

# Let's open the fits file and call it HDUL (Header Data Unit List, which is some historical name I guess)
hdul = fits.open(file_name)


# In[4]:


# Sometimes there are multiple FITS files in a single FITS file. 
# This one only has a single file, though, which appears to be a single image.
hdul.info()

# we can print the header 
# it's better to use "display" which prints things nicely :)
header = hdul[0].header 


# In[5]:


display(header)


# In[6]:


# we can treat the header as a labelled list, and get things from that list:
print(header['OBSNUM'])


# In[7]:


# let's grab the image data since that's the only thing1
image = hdul[0].data


# # Raw Image Plot

# In[8]:


# let's plot the image
plt.figure(figsize=[5,5])
im = plt.imshow(image)
plt.colorbar(im,fraction=0.046, pad=0.04,label='Counts [arb. units]')
plt.tight_layout()
plt.title('Ring Nebula',fontsize=20)
plt.show()


# In[9]:


low  = np.percentile(image, 20) 
high = np.percentile(image, 99.5)


# # Adding Some Fancy Visuals :)

# In[10]:


# let's plot the image
plt.figure(figsize=[5,5])
im = plt.imshow(image, vmin=low,vmax=high,norm=mpl.colors.LogNorm(),
          origin='lower',cmap='magma')
plt.colorbar(im,fraction=0.046, pad=0.04,label='Counts [arb. units]')
plt.tight_layout()
plt.title('Ring Nebula',fontsize=20)
plt.show()


# In[11]:


# let's plot the image
plt.figure(figsize=[5,5])
im = plt.imshow(image, vmin=low,vmax=high,norm=mpl.colors.LogNorm(),
          origin='lower',cmap='magma')
plt.colorbar(im,fraction=0.046, pad=0.04,label='Counts [arb. units]')
plt.xlim(325,625)
plt.ylim(400,700)
plt.tight_layout()
plt.title('Ring Nebula',fontsize=20)
plt.show()


# # Image Stretching and Scaling
# 
# See: https://docs.astropy.org/en/stable/visualization/normalization.html

# In[12]:


from astropy.visualization import (ZScaleInterval,MinMaxInterval, LogStretch, LinearStretch,
                                   ImageNormalize)


# In[13]:


# Create an ImageNormalize object
norm = ImageNormalize(image, interval=MinMaxInterval(),
                      stretch=LogStretch())

# or equivalently using positional arguments
# norm = ImageNormalize(image, MinMaxInterval(), SqrtStretch())

# Display the image
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image, origin='lower', norm=norm, cmap='Greys')
fig.colorbar(im)


# In[ ]:





# # An alternate way to read files that I really like, and you might see:

# In[14]:


with fits.open(file_name) as f:
    data = f[0].data
    f.close()


# In[15]:


plt.imshow(data)


# In[ ]:




