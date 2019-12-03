#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing pandas library 
import pandas as pd
data = pd.read_csv('D:\df3 (1).csv')

# importing the suprise file 
from surprise import Reader, Dataset
reader = Reader()
data = Dataset.load_from_df(data[['customer_id', 'product_id', 'star_rating']], reader)

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)


# In[3]:


#import the suprise library abd 
from surprise import SVD, accuracy
algo = SVD()
algo.fit(trainset)


# In[4]:


# predicting the output 
predictions = algo.test(testset)


# In[5]:


#Accuracy measures for the SVD model
from surprise import accuracy
accuracy.rmse(predictions)

