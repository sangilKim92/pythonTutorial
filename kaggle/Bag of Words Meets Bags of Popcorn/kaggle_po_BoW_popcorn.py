
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt

from collections import Counter

from wordcloud import WordCloud
import nltk

train = pd.read_csv("./labeledTrainData.tsv.zip", header=0, \
                    delimiter="\t", quoting=3)
test= pd.read_csv("./testData.tsv.zip", header=0, delimiter="\t", quoting=3)
print(train['id'].unique().size)
#id의 개수와 데이터의 개수 동일. id는 삭제해도 된다고 판단됩니다.).

target=train['sentiment']
train=train.drop('sentiment', axis=1)
train['review']=train['review'].apply(lambda x: x[1:len(x)])
train=train.drop('id', axis=1)
values=train.values
train=train['review'].apply(lambda x:re.sub("[^a-zA-Z]", " ",x))
#정규식을 이용해서 불용어 제거
temp=list(map(lambda x: x.lower().split(),train))


galexy_stop_words="it the to and in has he in a and I his It too The that this with on was of"
galexy_nouns = []
for post in train:
    for noun in nltk.word_tokenize(post): #문장을 명사로 바꾼다.
        if noun not in galexy_stop_words: #filter로 필요없는 명사만 걸러낸다.
            galexy_nouns.append(noun)

galexy_nouns[0:10]


num_top_nouns = 100
galexy_nouns_counter = Counter(galexy_nouns)
galexy_top_nouns = dict(galexy_nouns_counter.most_common(num_top_nouns))

galexy_wc = WordCloud(background_color="white",max_words=1000,
        max_font_size=40,
        scale=3,
        random_state=1 )
galexy_wc.generate_from_frequencies(galexy_top_nouns)
fig = plt.figure(1, figsize=(15, 15))
plt.axis('off')

plt.imshow(galexy_wc)
plt.show()