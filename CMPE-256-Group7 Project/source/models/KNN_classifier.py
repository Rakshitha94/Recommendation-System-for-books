#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import string
#from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error


# In[2]:


#read data from csv file
df=pd.read_csv(r'C:\Users\rishi\OneDrive\Desktop\fulldata.csv',index_col=0)


# In[3]:


print(df.columns)
print(df.shape)


# In[4]:


#get count and mean and groupby the similar product ids
count = df.groupby("product_id", as_index=False).count()
mean = df.groupby("product_id", as_index=False).mean()

dfMerged = pd.merge(df, count, how='right', on=['product_id'])
dfMerged


# In[5]:


#rename column
dfMerged["totalReviewers"] = dfMerged["review_id_y"]
dfMerged["overallScore"] = dfMerged["star_rating_x"]
dfMerged["summaryReview"] = dfMerged["review_body_x"]

dfNew = dfMerged[['product_id','summaryReview','overallScore',"totalReviewers"]]


# In[6]:


dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 100]
dfCount


# In[ ]:





# In[7]:


dfProductReview = df.groupby("product_id", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("product_id")["summaryReview"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.to_csv("ProductReviewSummary.csv")


# In[8]:


dfProductReview


# In[9]:


df3 = pd.read_csv("ProductReviewSummary.csv")
df3 = pd.merge(df3, dfProductReview, on="product_id", how='inner')


# In[10]:


df3 = df3[['product_id','summaryReview','star_rating']]


# In[11]:


#function for tokenizing summary
regEx = re.compile('[^a-z]+')
def cleanReviews(reviewText):
    reviewText = reviewText.lower()
    reviewText = regEx.sub(' ', reviewText).strip()
    return reviewText


# In[12]:


#reset index and drop duplicate rows
df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)
df3 = df3.drop_duplicates(['star_rating'], keep='last')
df3 = df3.reset_index()


# In[13]:


reviews = df3["summaryClean"] 
countVector = CountVectorizer(max_features = 300, stop_words='english') 
transformedReviews = countVector.fit_transform(reviews) 

dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)


# In[14]:



#save 
dfReviews.to_csv("dfReviews.csv")


# In[15]:


# First let's create a dataset called X
X = np.array(dfReviews)
 # create train and test
tpercent = 0.9
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)


# In[17]:



neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)


# In[18]:


#find most related products
for i in range(lentest):
    a = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = a[1]

    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    print ("Based on product reviews, for ", df3["product_id"][lentrain + i] ," average rating is ",df3["star_rating"][lentrain + i])
    print ("The first similar product is ", df3["product_id"][first_related_product] ," average rating is ",df3["star_rating"][first_related_product])
    print ("The second similar product is ", df3["product_id"][second_related_product] ," average rating is ",df3["star_rating"][second_related_product])
    print ("-----------------------------------------------------------")


# In[19]:


print ("Based on product reviews, for ", df3["product_id"][260] ," average rating is ",df3["star_rating"][260])
print ("The first similar product is ", df3["product_id"][first_related_product] ," average rating is ",df3["star_rating"][first_related_product])
print ("The second similar product is ", df3["product_id"][second_related_product] ," average rating is ",df3["star_rating"][second_related_product])
print ("-----------------------------------------------------------")


# In[20]:


#nearest neighbors=3
df5_train_target = df3["star_rating"][:lentrain]
df5_test_target = df3["star_rating"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

n_neighbors = 3
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)

print(classification_report(df5_test_target, knnpreds_test))


# In[21]:


print (accuracy_score(df5_test_target, knnpreds_test))


# In[22]:


print(mean_squared_error(df5_test_target, knnpreds_test))


# In[23]:


#nearest neighbors=5
df5_train_target = df3["star_rating"][:lentrain]
df5_test_target = df3["star_rating"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

n_neighbors = 5
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)
#print (knnpreds_test)

print(classification_report(df5_test_target, knnpreds_test))


# In[24]:


print (accuracy_score(df5_test_target, knnpreds_test))


# In[25]:


print(mean_squared_error(df5_test_target, knnpreds_test))


# In[26]:


# First let's create a dataset called X
X = np.array(dfReviews)
 # create train and test
tpercent = 0.85
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)


# In[27]:


# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.
neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)


# In[28]:


for i in range(lentest):
    a = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = a[1]

    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    print ("Based on product reviews, for ", df3["product_id"][lentrain + i] ," average rating is ",df3["star_rating"][lentrain + i])
    print ("The first similar book is ", df3["product_id"][first_related_product] ," average rating is ",df3["star_rating"][first_related_product])
    print ("The second similar book is ", df3["product_id"][second_related_product] ," average rating is ",df3["star_rating"][second_related_product])
    print ("-----------------------------------------------------------")


# In[29]:


df5_train_target = df3["star_rating"][:lentrain]
df5_test_target = df3["star_rating"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

n_neighbors = 5
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)
#print (knnpreds_test)

print(classification_report(df5_test_target, knnpreds_test))


# In[30]:


print (accuracy_score(df5_test_target, knnpreds_test))


# In[31]:


print(mean_squared_error(df5_test_target, knnpreds_test))


# In[ ]:


#using brute force knn algorithm


# In[32]:


neighbor = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(dfReviews_train)

distances, indices = neighbor.kneighbors(dfReviews_train)


# In[33]:


df5_train_target = df3["star_rating"][:lentrain]
df5_test_target = df3["star_rating"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)
n_neighbors = 3
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)

print(classification_report(df5_test_target, knnpreds_test))
print ("Accuracy: ",accuracy_score(df5_test_target, knnpreds_test))
print("MSE: ",mean_squared_error(df5_test_target, knnpreds_test))


# In[ ]:


# using kd-tree algorithm


# In[34]:


neighbor = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(dfReviews_train)
distances, indices = neighbor.kneighbors(dfReviews_train)


# In[35]:


df5_train_target = df3["star_rating"][:lentrain]
df5_test_target = df3["star_rating"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)
n_neighbors = 5
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(dfReviews_train, df5_train_target)
knnpreds_test = knnclf.predict(dfReviews_test)

print(classification_report(df5_test_target, knnpreds_test))
print ("Accuracy: ",accuracy_score(df5_test_target, knnpreds_test))
print("MSE: ",mean_squared_error(df5_test_target, knnpreds_test))


# In[ ]:




