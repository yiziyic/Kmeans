# Kmeans
------
K-Means是一种聚类算法，使用它可以为数据分类。K代表你要把数据分为几个组。  
K-means算法的基本思想是初始随机给定K个簇中心，按照最邻近原则把待分类样本点分到各个簇。然后按平均法重新计算各个簇的质心，从而确定新的簇心。一直迭代，直到簇心的移动距离小于某个给定的值。    
代码如下：

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

from matplotlib import pyplot
import xlrd

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
    
    def fit(self, data):
        self.centers_ = {}
        
        for i in range(self.k_):
            self.centers_[i] = data[i]
 
        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
 
            #print("质点:",self.centers_)
 
            for feature in data:
                #distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature-self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)
 
            #print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c],axis=0)
 
            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum( (cur_centers-org_centers)/org_centers*100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break
 
    def predict(self, p_data):
        distances = [np.linalg.norm(p_data-self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index
 
if __name__ == '__main__':
    # 加载数据
    df = pd.read_excel(r"D:\tt.xlsx",sheet_name=0)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)  # 把NaN替换为0
    #print(df.shape)  
    #print(df.head())
    #print(df.tail())
    #df.convert_objects(convert_numeric=True)
    #df.fillna(0, inplace=True) 
    x = np.array(df)
    x = preprocessing.scale(x,axis=0, with_mean=True, with_std=True, copy=True)

    #clf = KMeans(n_clusters=2)
    #clf.fit(x)
    k_means = K_Means(k=2)#k表示分为几类
    k_means.fit(x)

    print(k_means.centers_)
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*',s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
        #pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))
        #pyplot.scatter(point[0], point[1],point[2],point[3],point[4],c=('r' if cat == 0 'y' else if cat==1 'g' else if cat==2 'm' else if cat==3  'b' else if cat==4))        
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

```
执行结果：
![挂了吗](https://github.com/yiziyic/Kmeans/blob/master/aswecansee.jpg)


---------
先这样写啦，有空把算法讲清楚！
待解决问题：
- [ ] 分组还是不太好用，反正最后画图的点的颜色需要改一下
- [ ] 数据分析的结果能不能更具体哇  


最后偷偷感谢一下孙老师吧
