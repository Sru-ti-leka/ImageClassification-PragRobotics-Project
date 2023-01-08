#!/usr/bin/env python
# coding: utf-8

# # Image Processing Project by Srutileka S

# ## Importing libraries 

# In[1]:


import os 
import warnings
warnings.simplefilter('ignore')


# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.io import imread, imshow 
from skimage.transform import resize 
from skimage.color import rgb2gray
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# ## Importing working directory 

# In[3]:


#import working dir
me= os.listdir("C:/Users/lekas/Desktop/Image dataset/SrutiLeka")
je= os.listdir("C:/Users/lekas/Desktop/Image dataset/JacobElordi")
ss= os.listdir("C:/Users/lekas/Desktop/Image dataset/SydneySweeny")


# ## Reading images as matrix

# In[4]:


limit=10 
me_img=[None]*limit

j=0

for i in me:
    if(j<limit):
        me_img[j]=imread("C:/Users/lekas/Desktop/Image dataset/SrutiLeka/"+i)   
        j+=1
    else:
        break


# In[5]:


limit=10 
je_img=[None]*limit

j=0

for i in je:
    if(j<limit):
        je_img[j]=imread("C:/Users/lekas/Desktop/Image dataset/JacobElordi/"+i)   
        j+=1
    else:
        break


# In[6]:


limit=10
ss_img=[None]*limit

j=0 
        
for i in ss:
    if(j<limit):
        ss_img[j]=imread("C:/Users/lekas/Desktop/Image dataset/SydneySweeny/"+i)   
        j+=1
    else:
        break 


# ## Reading images in grayscale

# In[7]:


limit=10

me_gray=[None]*limit


j=0

for i in me:
    if(j<limit):
        me_gray[j]=rgb2gray(me_img[j])
        j+=1
    else:
        break


# In[8]:


limit=10

je_gray=[None]*limit


j=0

for i in je:
    if(j<limit):
        je_gray[j]=rgb2gray(je_img[j])
        j+=1
    else:
        break


# In[9]:


limit=10

ss_gray=[None]*limit


j=0

for i in ss:
    if(j<limit):
        ss_gray[j]=rgb2gray(ss_img[j])
        j+=1
    else:
        break


# ## Show image 

# In[10]:


imshow(me_gray[0])


# In[11]:


imshow(je_gray[0])


# In[12]:


imshow(ss_gray[0])


# ## Define length 

# In[13]:


len_of_me=len(me_gray)
len_of_je=len(je_gray)
len_of_ss=len(ss_gray)


# ## Resize all images 

# In[14]:


for i in range (10):
    m=me_gray[i]
    me_gray[i]=resize(m,(512,512))


# In[15]:


for i in range (10):
    j=je_gray[i]
    je_gray[i]=resize(j,(512,512))


# In[16]:


for i in range (10):
    s=ss_gray[i]
    ss_gray[i]=resize(s,(512,512))


# ## Flatten images 

# In[17]:


image_size_me=me_gray[0].shape
image_size_je=je_gray[0].shape
image_size_ss=ss_gray[0].shape

flatten_me= image_size_me[0]*image_size_me[1]
flatten_je= image_size_je[0]*image_size_je[1]
flatten_ss= image_size_ss[0]*image_size_ss[1]


# In[18]:


for i in range (len_of_me):
    me_gray[i]=np.ndarray.flatten(me_gray[i]).reshape(flatten_me,1)


# In[19]:


for i in range (len_of_je):
    je_gray[i]=np.ndarray.flatten(je_gray[i]).reshape(flatten_je,1)


# In[20]:


for i in range (len_of_ss):
    ss_gray[i]=np.ndarray.flatten(ss_gray[i]).reshape(flatten_ss,1)


# In[21]:


me_gray=np.dstack(me_gray)
je_gray=np.dstack(je_gray)
ss_gray=np.dstack(ss_gray)


# In[22]:


me_gray=np.rollaxis(me_gray, axis=2, start=0)
je_gray=np.rollaxis(je_gray, axis=2, start=0)
ss_gray=np.rollaxis(ss_gray, axis=2, start=0)


# In[23]:


me_gray=me_gray.reshape(len_of_me,flatten_me)
je_gray=je_gray.reshape(len_of_je,flatten_je)
ss_gray=ss_gray.reshape(len_of_ss,flatten_ss)


# ## Create Data frames

# In[24]:


me_data=pd.DataFrame(me_gray)
je_data=pd.DataFrame(je_gray)
ss_data=pd.DataFrame(ss_gray)


# In[25]:


me_data["label"]="Sruti Leka"
je_data["label"]="Jacob Elordi"
ss_data["label"]="Sydney Sweeny"


# In[26]:


me_data


# ## Concat into single data frame 

# In[27]:


celebs=pd.DataFrame()
celebs=celebs.append(me_data, ignore_index = True)
celebs=celebs.append(je_data, ignore_index = True)
celebs=celebs.append(ss_data, ignore_index = True)


# In[28]:


celebs


# ## Shuffle dataset 

# In[29]:


celebs_indexed = shuffle(celebs).reset_index()


# In[30]:


celebs=celebs_indexed.drop(["index"],axis=1)


# In[31]:


celebs


# ## SVM Classification 

# ### Assigning dependent and independednt variables 

# In[32]:


x=celebs.values[:,:-1]
y=celebs.values[:,-1]


# ### Splitting Dataset

# In[33]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)


# ### Training Dataset

# In[34]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# ### Prediction 

# In[35]:


y_pred=clf.predict(x_test)
y_pred


# ### Visialization 

# In[36]:


for i in (np.random.randint(0,6,4)):  
    predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title("predicted label:{0}".format(y_pred[i]))
    plt.imshow(predicted_images,interpolation="nearest", cmap="gray")
    plt.show()


# ### Check Accuracy 

# In[37]:


accuracy = metrics.accuracy_score(y_test,y_pred)


# In[38]:


accuracy


# In[39]:


confusion_matrix(y_test, y_pred)


# In[ ]:




