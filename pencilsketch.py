#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import plotly.express as px


# In[54]:


import sys

# Install required packages
get_ipython().system('{sys.executable} -m pip install opencv-python numpy plotly')

import cv2
import numpy as np
import plotly.express as px

# Loading Image
img = cv2.imread("dq.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Displaying Image
imgs = px.imshow(img)
imgs.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
imgs.show()


# In[55]:


#Resizing image shape
scale_percent = 0.60
width = int(img.shape[1]*scale_percent)
height = int(img.shape[0]*scale_percent)
dim = (width,height)
resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
res=px.imshow(resized)
res.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
res.show()


# In[56]:


#Sharpening Image
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
sharpened = cv2.filter2D(resized,-1,kernel_sharpening)
sharp=px.imshow(sharpened)
sharp.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
sharp.show()


# In[57]:


#Converting an image into gray_scale image
grayscale = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
gray = px.imshow(grayscale, color_continuous_scale='gray')
gray.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
gray.show()


# In[58]:


#Inverting the image
invs = 255-grayscale
inv=px.imshow(invs,color_continuous_scale='gray')
inv.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
inv.show()


# In[59]:


#Inverting the image
invs = 255-grayscale
inv=px.imshow(invs,color_continuous_scale='gray')
inv.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
inv.show()


# In[60]:


#Inverting the image
invs = 255-grayscale
inv=px.imshow(invs,color_continuous_scale='gray')
inv.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
inv.show()


# In[ ]:




