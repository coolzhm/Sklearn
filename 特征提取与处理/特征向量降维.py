'''
2)特征向量降维基本方法是
   1).单词全部转换成小写
   2).去掉文集常用词，即停用词(stop-word)像a，an，the，助动词do，be，will，介词on，around，beneath等。
   停用词通常是构建文档意思的功能词汇，其字面意义并不体现。CountVectorizer类可以通过设置stop_words参数过滤停用词，
   默认是英语常用的停用词。
   3).词跟还原与词形还原
'''

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

'''
#词跟还原(stemming)与词形还原(lemmatization),可以用nltk的wordNetLemmatizer实现  
#nltk.stemmer.porter.PorterStemmer 类是一个用于从英文单词中获得符合语法的（前缀）词干的极其便利的工具.  
'''
# 例子
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

wordnet_tags = ['n', 'v']
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
stemmer = PorterStemmer()
print('Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])


# 词形还原
def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token


lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print(tagged_corpus)
print('Lemmatized', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
