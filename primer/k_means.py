# Python3
# author:allenyzx
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from math import sqrt,pow



class KMeans:


    def __init__(self,kseed,times = 100,ktype='normal'):
        """
        :param kseed: 标签数目 int
        :param times: 迭代次数 int
        :param ktype: 算法种类 'normal' '+'
        """
        self.kseed = kseed
        self.times = times
        self.seed = []
        # +
        self.ktype = ktype

    # 随机种子
    def random_seed(self,rows):
        seed_array = np.random.uniform(low=0,high=10,size=(self.kseed,rows))
        return seed_array


    def _new_calc_k_seed(self,X):
        key  = np.random.randint(low=0,high=len(X))
        step = 0
        seed_list = []

        while step < self.times:
            if step == 0:
                seed = X[key]
            else:
                distance_list = []
                for x in X :
                    distance = sum(map(lambda x: sqrt(pow(x, 2)), (seed - x)))
                    distance_list.append(distance)
                dx = sum(distance_list)
                random = np.random.random() *dx
                for i in distance_list:
                    random -= i
                    if random<=0:
                        seed = i
                        break
                x_index = distance_list.index(seed)
                seed = X[x_index]
                seed_list.append(seed)

                if len(seed_list) ==self.kseed:
                    break
            step += 1

        seed_list = np.asarray(seed_list)

        return seed_list



    # 计算分类
    def _calcClassify(self,seed):
        sort_array = []
        X_index = 0
        for x in X:
            distance_list = []
            for s in seed:
                # 欧氏距离
                distance = sum(map(lambda x: sqrt(pow(x, 2)), (s - x)))
                distance_list.append(distance)
            distance_list.insert(0, X_index)
            sort_array.append(distance_list)
            X_index += 1
        seed_array = []
        for x in sort_array:
            a = x[1:].index(min(x[1:]))
            seed_array.append([a]+X[x[0]].tolist())
        seed_array = np.asarray(seed_array)
        ### ([label, [X]])
        return seed_array

    # 创建新种子
    def _calcNewSeed(self,seed,seed_array):
        new_seed = []
        for s_index in range(self.kseed):
            X_array = seed_array[seed_array[:, 0] == s_index][:, 1:]
            if len(X_array) > 0:
                new_seed.append(np.sum(X_array, axis=0) / len(X_array))
            else:
                new_seed.append(seed[s_index])
        new_seed = np.asarray(new_seed)
        return new_seed




    def fit(self,X):
        rows = len(X[0])
        step = 0

        # 迭代次数
        while step < self.times:
            if step == 0:
                if self.ktype == 'normal':
                    seed = self.random_seed(rows=rows)
                elif self.ktype == '+':
                    seed =self._new_calc_k_seed(X)
                else:
                    raise ValueError('ktype set error')
            else:
                # print("input is %s" % seed)
                last_seed = seed
                seed_array = self._calcClassify(seed)
                seed = self._calcNewSeed(seed,seed_array)
                # print("output is %s" % seed)

                # 上次种子和这次种子总和相同，迭代结束
                if np.sum(np.asarray(last_seed)) == np.sum(np.asarray(seed)):
                    print("step is %s"%step)
                    break

            step += 1

        self.seed =seed

    def predict(self,X):
        return self._calcClassify(self.seed)




def plot_with_color(X,kseed=3):

    y = np.reshape(X[:,0],newshape=(len(X),1))

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    array = TSNE(random_state=999).fit_transform(X[:,1:])

    merge_array = np.concatenate((y,array),axis=1)

    color  = {0:'red',1:'green',2:'blue'}

    for i in range(kseed):
        kidArray = merge_array[merge_array[:,0]==i]
        plt.scatter(kidArray[:,1],kidArray[:,2],color=color[i],marker='o')

    plt.show()


if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target


    # kmean = KMeans(3,times=200,ktype='+')
    # kmean.fit(X)
    # array = kmean.predict(X)
    # plot_with_color(array)
    
    # sklearn
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=3)
    clf.fit(X)
    clf_y = clf.predict(X)
    clf_y = np.reshape(clf_y,newshape=(len(X),1))
    merge_array = np.concatenate((clf_y,X),axis=1)
    plot_with_color(merge_array)
