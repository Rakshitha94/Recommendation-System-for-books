#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Read the data, skip any lines that return an error
reviews = pd.read_csv(
    r'C:\Users\rishi\OneDrive\Desktop\amazon_reviews_us_Books_v1_02.tsv',
    sep='\t',
    error_bad_lines=False,
    warn_bad_lines=False)
reviews.head(3)


# In[2]:


reviews.info()


# In[ ]:





# In[3]:


#check for duplicates, and drop if any
sum(reviews.review_id.duplicated())


# In[4]:


purchase_ids = ['customer_id', 'product_id']

# Get a dataframe consisting only of reviews that are duplicated
duplicates = reviews[reviews.duplicated(subset=purchase_ids,
                                        keep=False)].sort_values(purchase_ids)
duplicates.head(4)


# In[5]:


reviews = (reviews
           # Sort the values so we'll keep the most recent review.
           .sort_values(['customer_id', 'product_id', 'review_date'], ascending=[False, False, True])
           .drop_duplicates(subset=purchase_ids, keep='last'))


# In[6]:


reviews.product_title.value_counts().to_frame().head(5)
reviews.product_title.value_counts().to_frame().tail(5)


# In[7]:


reviews[['product_parent',
         'product_id']].drop_duplicates().product_parent.value_counts().head(5)


# In[8]:


reviews[reviews.product_parent == 43217624][[
    'product_parent', 'product_id', 'product_title'
]].drop_duplicates().head()


# In[9]:


reviews[reviews.product_parent == 43217624][['product_title'
                                              ]].drop_duplicates()


# In[10]:


products = reviews[['product_id', 'product_title']].drop_duplicates(
    subset='product_title', keep='first')
column_order = reviews.columns
reviews = reviews.drop(
    'product_id', axis=1).merge(
        products, on='product_title')[column_order]

reviews[['product_id', 'product_title'
         ]].drop_duplicates().product_title.value_counts().head(5).to_frame()


# In[11]:


reviews[['product_parent',
         'product_id']].drop_duplicates().product_parent.value_counts().head(5)


# In[12]:


reviews[reviews.product_parent == 43217624][['product_title'
                                              ]].drop_duplicates().head()

reviews[reviews.product_parent == 669379389][['product_title'
                                              ]].drop_duplicates().head()


# In[13]:


freq=(reviews.customer_id.value_counts().rename_axis('id').reset_index(
    name='frequency').frequency.value_counts(
        normalize=False).rename_axis('reviews').to_frame().head(10))


# In[14]:


freq


# In[15]:


reviews.shape


# In[16]:


reviews=reviews[['customer_id','review_id','product_id','product_parent','product_title','star_rating','review_headline']]


# In[17]:


df=reviews


# In[18]:


df.shape


# In[19]:


print(df.columns)
print(df.shape)


# In[20]:


count = df.groupby("product_id", as_index=False).count()
mean = df.groupby("product_id", as_index=False).mean()

dfMerged = pd.merge(df, count, how='right', on=['product_id'])
dfMerged


# In[21]:


dfMerged = pd.merge(df, count, how='right', on=['product_id'])
dfMerged


# In[22]:


#rename column
df["totalReviewers"] = df["customer_id"]
df["overallScore"] = df["star_rating"]
df["summaryReview"] = df["review_headline"]

dfNew = df[['product_id','product_title','summaryReview','overallScore',"totalReviewers"]]


# In[23]:


dfNew


# In[30]:


len(dfNew)


# In[31]:


df = dfNew.sort_values(by='totalReviewers', ascending=False)
dfCount = dfNew[dfNew.totalReviewers >= 100]
dfCount


# In[42]:


dfCount['star_rating']=dfCount['overallScore']
dfCount['customer_id']=dfCount['totalReviewers']
data=dfCount


# In[48]:


data1=data[['product_id','product_title','star_rating','customer_id','summaryReview']]
#save to csv file
data1.to_csv('final_data.csv')

