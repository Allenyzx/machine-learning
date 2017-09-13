# Python3
# author:allenyzx
import operator
from math import log
import time


# 数据集
def datasets():
    dataSet=[[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels = ['feature1','feature2']
    return dataSet, labels


# 计算最大熵
def calcEnt(X):
    numEntries = len(X)
    labelsCounts = {}
    for feaVec in X:
        currentLabel = feaVec[-1]
        if currentLabel not in labelsCounts:
                labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel]+=1

    shannonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)

    return shannonEnt

# 分割数据集
def splitDataset(X,axis,value):
    retDataSet = []
    # 轮询每一行
    for featVec in X:
        # 如果每一行所对应的最优特征的特征值==目标特征值
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            # 取该目标特征值的数据集，且删除最优特征那一列
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    # 返回取该目标特征值的数据集，且删除最优特征那一列。
    return retDataSet


# 选择最好的属性分割
def chooseBestFeatureToSplit(X):
    numFeatures = len(X[0]) - 1
    baseEntroy = calcEnt(X)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in X]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataset(X,i,value)
            prob = len(subDataSet)/float(len(X))
            newEntropy += prob*calcEnt(subDataSet)
        infoGain = baseEntroy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

# 但子集剩下特征值为一个时，多数投票表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0

        classCount[vote]+=1

    return max(classCount)


# 创建树
def createTree(X,labels):
    classList = [example[-1] for example in X]
    # 如果树的标签数量等于标签列表长度，说明只有一个标签，树无法切割
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果特征值的数量只有一种，那么选标签数最多的那一类
    if len(X[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的特征属性索引
    bestFeat = chooseBestFeatureToSplit(X)
    # 选择最优的特征属性
    bestFeatLabel = labels[bestFeat]
    # 创建树结构
    myTree ={bestFeatLabel:{}}
    # 删除该特征
    del labels[bestFeat]
    # 选择最优的特征的那一列
    featValues = [example[bestFeat] for example in X]
    # 取对应最优特征的特征值集合
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 新的特征名称列表
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataset(X,bestFeat,value),subLabels)

    return myTree


# 主进程
def main():
    data, label = datasets()
    print(data)
    myTree = createTree(data, label)
    print(myTree)











if __name__ == '__main__':

    datset,labels = datasets()
    print(datset)
    main()
