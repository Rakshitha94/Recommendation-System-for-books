#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import os


# In[2]:


#load data from csv file
df=pd.read_csv(r'books1.csv',index_col=0)


# In[3]:


df.head()


# In[4]:


len(df)


# In[5]:


df=df.head(50000)


# In[ ]:





# In[6]:


df.info()


# In[7]:


df=df.fillna('')


# In[8]:


#data splitting into train and test data
import numpy as np
threshold=np.random.rand(len(df))<0.8
train_data=df[threshold]
test_data=df[~threshold]


# In[9]:


train_data


# In[10]:


test_data


# In[11]:


data=train_data.groupby(['product_id']).agg(lambda x:list(x)).reset_index()


# In[12]:


data


# In[13]:


data_1=data


# In[14]:


data_1['product_title']=data_1['product_title'].apply(lambda x: x[:1])


# In[15]:


data_1


# In[16]:


data_1=data_1.applymap(str).reset_index()


# In[17]:


def create_soup(x):
    return ' '.join(x['review_body'])+' '.join(x['review_id'])+' '.join(x['review_headline'])


# In[18]:


data_1['soup']=data_1.apply(create_soup,axis=1)


# In[19]:


#data_1=data_1.applymap(str).reset_index()


# In[20]:


#tf idf vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words='english')
cm=tf.fit_transform(data_1['soup'])
cm.shape


# In[21]:


print("calculating similarity")
from sklearn.metrics.pairwise import cosine_similarity
cos_sim=cosine_similarity(cm,cm)
print("calculated..yaaay")


# In[22]:


i=pd.Series(data_1.index,index=data_1['product_title']).drop_duplicates()


# In[30]:


def content_recommender(product_title, cosine_sim_is=cos_sim, df=data_1, indices=i):
    
    idx = i[product_title]

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
    recommender_metadata.to_csv(" 3.csv", index=False)


# In[31]:


content_recommender("['State Of Fear']")


# In[32]:


content_recommender("['The Next Killer App']")


# In[33]:


content_recommender("['Carolans Farewell']")


# In[40]:


y = df[df['product_title'].astype(str).str.contains("Carolans Farewell")]


# In[41]:


y


# In[39]:


df[df['customer_id']=='53007445']['product_title'].values


# In[ ]:




