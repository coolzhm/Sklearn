'''
线性回归

'''

'''
一元线性回归
假设你想计算匹萨的价格。虽然看看菜单就知道了，不过也可以用机器学习方法建一个线性回归模型，
通过分析匹萨的直径与价格的数据的线性关系，来预测任意直径匹萨的价格。
我们先用scikit-learn写出回归模型，然后我们介绍模型的用法，以及将模型应用到具体问题中。
假设我们查到了部分匹萨的直径与价格的数据，这就构成了训练数据，如下表所示：

训练样本	直径（英寸）	价格（美元）
1	6	7
2	8	9
3	10	13
4	14	17.5
5	18	18
'''
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 这个属性设置是让matplot画图时显示中文的标签
font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=15)


# 定义画图函数
def runplt():
    plt.figure()
    plt.title('披萨价格与直径数据', fontproperties=font)
    plt.xlabel('直径(英寸)', fontproperties=font)
    plt.ylabel('价格(美元)', fontproperties=font)
    plt.axis([0, 25, 0, 25], fontproperties=font)
    plt.grid(True)
    return plt


# 训练集数据
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

# 导入一元线性回归函数:y = α + βx
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)  # 训练集数据放入模型中
print('预测一张12英寸披萨价格：$%.2f' % model.predict([[12]]))

plt = runplt()
X2 = [[0], [10], [14], [25]]
y2 = model.predict(X2)  # 预测数据
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')

# 残差预测值
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')

plt.show()