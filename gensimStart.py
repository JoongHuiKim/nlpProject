from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import word_tokenize


def clean(char):
    text = []
    filter = '~!@#$%^&*()=+[{]}:?,<>/unspecified'
    filter2 = ''
    for letter in char:
        if letter in filter:
            continue
        else:
            if letter in filter2:
                continue
            else:
                text.append(letter)
    return text


fileName = 'C:/Users/amc/Documents/my/research2.xlsx'

file = pd.read_excel(fileName)
list_file = list(file['STD_DIAG_NM'])
l_list_file = []
for i in range(len(list_file)):
    sen = list_file[i]
    l_list_file.append(sen.lower())

list_file = []
list_file = l_list_file

for k in range(5):
    print(word_tokenize(list_file[k]))


list_token = [[] for _ in range(len(list_file))]

for i in range(len(list_file)):
    list_txt = word_tokenize(list_file[i])
    txt1 = clean(list_txt)
    for txt2 in txt1:
        list_token[i].append(txt2)

print(list_token)

model = Word2Vec(sentences=list_token, vector_size=156, window=5, min_count=1, workers=4,sg = 0)

for idx,word in enumerate(model.wv.index_to_key):
    if idx == 30:
        break
    print(word)

vec_corona = model.wv['covid-19']
print(vec_corona.shape)
print(vec_corona)


for i,(word,similarity) in enumerate(model.wv.most_similar(positive=['covid-19'],topn= 20)):
    print(word,similarity)

for i,(word,similarity) in enumerate(model.wv.most_similar(positive=['cerebral','liver'], negative= ['nerves'],topn= 20)):
    print(word,similarity)