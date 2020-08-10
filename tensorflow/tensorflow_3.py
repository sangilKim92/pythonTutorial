#한 줄단위로 실행하려면 alt + shitf + e
import os
import re
import tensorflow as tf
import pandas as pd
from tensorflow.keras import utils
import matplotlib.pyplot as plt

'''
data_set = tf.keras.utils.get_file(
    fname="imdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True #압축풀기 true
)
'''

def directory_data(directory):
    data ={}
    data['review'] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory,file_path),"r", encoding='UTF8') as file:
            data['review'].append(file.read())

    return pd.DataFrame.from_dict(data)

def data(directory):
    pos_df = directory_data(os.path.join(directory,"pos"))
    neg_df = directory_data(os.path.join(directory,"neg"))
    pos_df['sentiment'] = 1
    neg_df['sentiment'] = 0

    return pd.concat([pos_df,neg_df])

def to_csv(df,name):
    df.to_csv('./data/' + str(name)+'.csv', sep=',')


#train_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "train"))
#test_df = data(os.path.join(os.path.dirname(data_set),"aclImdb","test"))

#to_csv(train_df,'train')
#to_csv(test_df,'test')
train_df=pd.read_csv('./data/train.csv')
test_df= pd.read_csv('./data/test.csv')

print(train_df.head())

reviews = list(train_df['review'])

tokenized_reviews=[r.split() for r in reviews]

review_len_by_token = [len(t) for t in tokenized_reviews]

review_len_by_eumjeol = [len(s.replace(' ','')) for s in reviews]

plt.figure(figsize=(12,5))

plt.hist(review_len_by_token, bins=50, alpha= 0.5, color ='r', label='word')
plt.hist(review_len_by_eumjeol, bins=50, alpha = 0.5, color='b', label='alphabet')
plt.yscale('log', nonposy='clip')

plt.title('Review Length Histogram')

plt.xlabel('Review Length')
plt.ylabel('Number of Reviews')
import numpy as np

print('문장 최대 길이: {}'.format(np.max(review_len_by_token)))
print('문장 최소 길이: {}'.format(np.min(review_len_by_token)))
print('문장 평균 길이: {:.2f}'.format(np.mean(review_len_by_token)))
print('문장 길이 표준편차: {:.2f}'.format(np.std(review_len_by_token)))
print('문장 중간 길이: {}'.format(np.median(review_len_by_token)))
print('제1사분위 길이: {}'.format(np.percentile(review_len_by_token,25)))
print('제3사분위 길이: {}'.format(np.percentile(review_len_by_token,75)))

plt.figure(figsize=(12,5))

plt.boxplot([review_len_by_token],labels=['token'],showmeans=True)

plt.figure(figsize=(12,5))
plt.boxplot([review_len_by_eumjeol],labels=['Eumjeol'],showmeans=True)

from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black',width=800, height=600).generate(' '.join(train_df['review']))

plt.figure(figsize=(12,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

import seaborn as sns

sentiment = train_df['sentiment'].value_counts()
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6,3)
sns.countplot(train_df['sentiment'])