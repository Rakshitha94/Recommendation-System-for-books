#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import os


# In[2]:


#load data from csv file
df=pd.read_csv(r'final_data.csv',index_col=0)


# In[3]:


df.head()


# In[4]:


len(df)


# In[5]:


df=df.head(50000)


# In[7]:


df.info()


# In[8]:


df=df.fillna('')


# In[9]:


import numpy as np
threshold=np.random.rand(len(df))<0.8
train_data=df[threshold]
test_data=df[~threshold]


# In[10]:


train_data


# In[11]:


test_data


# In[12]:


# group similar items with same item ids
data=train_data.groupby(['product_id']).agg(lambda x:list(x)).reset_index()


# In[13]:


data


# In[14]:


data_1=data


# In[15]:


data_1['product_title']=data_1['product_title'].apply(lambda x: x[:1])
# data_1['review_id']=data_1['review_id'].apply(lambda x: x[:3])
# data_1['review_headline']=data_1['review_headline'].apply(lambda x: x[:3])
# data_1['review_body']=data_1['review_body'].apply(lambda x: x[:3])


# In[16]:


data_1


# In[17]:


def create_soup(x):
#     return ' '.join(x['review_body'])+' '.join(x['review_id'])+' '.join(x['review_headline'])
    return ' '.join(x['summaryReview'])


# In[18]:


data_1['soup']=data_1.apply(create_soup,axis=1)


# In[19]:


data_1=data_1.applymap(str).reset_index()


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(stop_words='english')
cm=tf.fit_transform(data_1['soup'])
cm.shape


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data_1['soup'])
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[22]:


print("calculating similarity")
from sklearn.metrics.pairwise import cosine_similarity
cos_sim=cosine_similarity(cm,cm)
print("calculated..yaaay")


# In[23]:


i=pd.Series(data_1.index,index=data_1['product_id']).drop_duplicates()


# In[28]:


def item_based_recommender(product_id, cosine_sim_is=cos_sim, df=data_1, indices=i):
    
    idx = i[product_id]

    # Get the pairwsie similarity scores of all users with that user
    # And convert it into a list of tuples
    sim_scores = list(enumerate(cosine_sim_is[idx]))

    # Sort the users based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar users. Ignore the first one.
    sim_scores = sim_scores[1:11]

    # Get the user indices
    user_indices = [i[0] for i in sim_scores]

    book =[]
    score = []
    index=[]
    book = data_1['product_title'].iloc[user_indices]
    index,score = map(list,zip(*sim_scores))
    recommender_metadata=pd.DataFrame({'Books' : book, 'Score' : score})
    print(recommender_metadata)
    #score = pd.DataFrame(score)
    #score=pd.DataFrame({'Score' : score,'Probability' : list2})
    #score['val'] = score
    #score.set_index('val')
    #yes['title'] = yes
    #yes['score'] = score
    recommender_metadata.to_csv(" 1.csv", index=False)


# In[29]:


a=item_based_recommender('0975599518')


# In[30]:


a=item_based_recommender('0889104689')


# In[ ]:


# Evaluation
test_data[test_data['customer_id'].astype(str).str.contains('53096584')]


# In[ ]:


a=user_based_recommender('a=user_based_recommender('0889104689')')

