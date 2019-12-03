# Recommendation-System-for-books
Our project book recommendations system, recommends books to the readers depending upon the ratings and reviews given by them.
The different approaches used are:
1. Content Based
2. Collaborative Filtering- User Based
                          - Item based
   and, K Nearest Neighbor Approach for CF
   
## Dataset:

The Amazon review dataset is selected for the recommendation model.\
It contains a collection of reviews and ratings from 1995 until 2015 (130M+ customer ratings).\
The dataset is in Tab separated Value(TSV), a text format and is  around 3GB memory size.\
Link for Dataset:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_02.tsv.gz


## Software Requirements:
1.Anaconda distribution\
2.Python 3\
3.pandas\
4.numpy\
5.sklearn\
6.scikit learn



To run the python files on HPC, use the following steps:\
module load python3\
python filename.py\
The files can be run on the local machine using jupyter notebook, but requires a huge memory and is time consuming.


The  preprocess folder contains the python file for preprocessing of data.\
The models folder contains the python files of content based, userbased nd item based collaborating filtering.\
The preprocessing file gives the output, final_data.csv, which is used for the models.\
The different models can be ran using the same command: python filename.py\
The output of the various models are stored in the .csv files, which are the top recommendations with their similarity scores.
