
# coding: utf-8

# #run all בתיבה למטה יש לשים לב להוראות ואז לעשות 

# In[1]:

#save small_net_4X3X3.net at your notebook. directory example: WinPython-64bit-2.7.10.2/notebooks/my notebook
import_loaction='F:/Users/Nadav/OneDrive/WinPython-64bit-2.7.10.2/notebooks/docs/Siduri - images for project 2/' #enter import folder
export_location='F:/Users/Nadav/OneDrive/WinPython-64bit-2.7.10.2/notebooks/docs/Siduri - images for project 2_Fixed_By_Net2/' #enter destination folder


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import copy
get_ipython().magic(u'matplotlib inline')
#%matplotlib qt
from PIL import Image
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.shortcuts import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pickle
import os


# In[3]:

images=[]
location=import_loaction
files = os.walk(location).next()[2]
file_count = len(files)
for k in range(file_count):
    name=str(k)+'.JPG'
    img = Image.open(location+name)
    images.append(img)
small_images = copy.deepcopy(images)
for i in range(len(small_images)):
    cols,rows=small_images[i].size
    if cols>rows:
        small_images[i]=small_images[i].resize((4, 3), Image.ANTIALIAS)
    else:
        small_images[i]=small_images[i].resize((3, 4), Image.ANTIALIAS)
for i in range(len(small_images)):
    small_images[i]=np.array(small_images[i])
images_arr=[]
for i in range(len(images)):
    images_arr.append(np.array(images[i]))
small_images=np.array(small_images)
small_images_list=[]
for i in range(len(small_images)):
    small_images_list.append(small_images[i].reshape(36))
small_images=small_images_list
total_small_images=np.vstack((small_images))
small_images_inputs=np.column_stack((range(len(total_small_images)),total_small_images))
small_indexes=small_images_inputs[:,0]
f=open('small_net_4X3X3.net','r')
small_net=pickle.load(f)
small_h=[]
for i in range(total_small_images.shape[0]):
    small_h.append(small_net.activate(total_small_images[i]))
small_h=np.array(small_h)
small_h_list=[]
for i in range(small_h.shape[0]):
    small_h_list.append(np.where(small_h[i]==max(small_h[i]),1,0))
small_h_arr=np.array(small_h_list)
k=0
location=export_location
if not os.path.exists(location):
    os.makedirs(location)
for i in small_h_arr:
    name=str(k)+'.jpg'
    if sum(abs(i-[1,0,0,0]))==0:
        images[k].save(location+name)
        k+=1
    elif sum(abs(i-[0,1,0,0]))==0:
        images[k].rotate(90).save(location+name)
        k+=1
    elif sum(abs(i-[0,0,1,0]))==0:
        images[k].rotate(180).save(location+name)
        k+=1
    else:
        images[k].rotate(270).save(location+name)
        k+=1

