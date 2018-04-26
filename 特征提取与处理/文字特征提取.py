'''
2.机器学习问题中常见的文档特征向量。
1)文字特征提取-词库模型（Bag-of-words model）：文字模型化最常用方法，可以看成是独热编码的一种扩展，
它为每个单词设值一个特征值。依据是用类似单词的文章意思也差不多。可以通过有限的编码信息实现有效的文档分类和检索。

CountVectorizer 类会将文档全部转换成小写，然后将文档词块化(tokenize).文档词块化是把句子分割成词块（token）
或有意义的字母序列的过程。词块大多是单词，但是他们也可能是一些短语，如标点符号和词缀。CountVectorizer类通过
正则表达式用空格分割句子，然后抽取长度大于等于2的字母序列。
'''
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())

print(vectorizer.vocabulary_)
'''
输出:
{'game': 3, 'ate': 0, 'unc': 9, 'duke': 2, 'the': 8, 'in': 4, 'lost': 5, 'basketball': 1, 'sandwich': 7, 'played': 6}
通过CountVectorizer类可以得出上面的结果。词汇表里面有10个单词，
但a不在词汇表里面，是因为a的长度不符合CountVectorizer类的要求。
'''

'''
对比文档的特征向量，会发现前两个文档相比第三个文档更相似。
如果用欧氏距离（Euclidean distance）计算它们的特征向量会比其与第三个文档距离更接近。

两向量的欧氏距离就是两个向量欧氏范数（Euclidean norm）或L2范数差的绝对值：d=||x0-x1||

向量的欧氏范数是其元素平方和的平方根：
scikit-learn里面的euclidean_distances函数可以计算若干向量的距离，
表示两个语义最相似的文档其向量在空间中也是最接近的。
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))