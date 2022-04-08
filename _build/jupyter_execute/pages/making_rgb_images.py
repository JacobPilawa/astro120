#!/usr/bin/env python
# coding: utf-8

# # Making RGB Images

# In[1]:


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
import matplotlib as mpl


# In[2]:


def plot_subimages(rgb):
    cmaps = ['Reds','Blues','Greens']

    fig, axes = plt.subplots(nrows=1,ncols=3,dpi=200)
    for c, col, ax in zip(rgb, cmaps, axes.flat):
        # get vmin, vmax 
        lo, hi = np.percentile(c[:], (20,99))
        ax.imshow(c.T,cmap=col,norm=mpl.colors.LogNorm(vmin=lo,vmax=hi),origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def load_image(r_filename,g_filename,b_filename):
    # read data
    image_r = fits.open(r_filename)[0].data
    image_g = fits.open(g_filename)[0].data
    image_b = fits.open(b_filename)[0].data
    # stack
    rgb = (np.dstack((image_r,image_g,image_b)))
    
    return rgb.T

def rescale(rgb, intensity_range):
    # initialize new_out
    rgb_rescaled = np.zeros(rgb.shape)
    
    for i, c in enumerate(rgb):
        lo, hi = np.percentile(c[:], (intensity_range[0],intensity_range[1]))
        rgb_rescaled[i] = rescale_intensity(c, in_range=(lo, hi))
    return rgb_rescaled.T


# In[3]:


r_filename = 'M16rA.fits'
g_filename = 'M16gA.fits'
b_filename = 'M16uA.fits'

rgb = load_image(r_filename,g_filename,b_filename)


# In[4]:


plot_subimages(rgb)


# In[5]:


# scale the values to be within the same percentiles
rgb_rescaled = rescale(rgb,[25,99.])


# In[6]:


plt.figure(figsize=[4,4],dpi=125)
plt.imshow(rgb_rescaled, origin='lower')
plt.show()


# In[ ]:





# In[ ]:




