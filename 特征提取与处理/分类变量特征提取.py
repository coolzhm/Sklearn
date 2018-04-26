'''
1、分类变量特征提取:分类数据的独热编码方法，并用scikit-learn的DictVectorizer类实现
许多机器学习问题都有分类的、标记的变量，不是连续的。例如，一个应用是用分类特征比如工作地点来预测工资水平。
分类变量通常用独热编码（One-of-K or One-Hot Encoding），通过二进制数来表示每个解释变量的特征。
'''

from sklearn.feature_extraction import DictVectorizer

onehot_encoder = DictVectorizer()
instances = [{'city': 'New York'}, {'city': 'San Francisco'}, {'city': 'Chapel Hill'}]
print(onehot_encoder.fit_transform(instances).toarray())
