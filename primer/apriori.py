# Python3
# author:allenyzx
import itertools
from collections import OrderedDict



def load_dataset():


    return [['a', 'c', 'd'], ['b', 'c', 'e'], ['a', 'b', 'c', 'e'], ['b', 'e'],['a','v'],['c','d']]



def calcFrequent(data):
    frequent = {}


    def recursion_create(subset2):

        # 如果频繁集只有一个，return结果
        if len(subset2)<=1:
            return

        #新的频繁集
        subset = {}
        # 旧的频繁集的长度
        number = len(subset2[0])

        # 将旧的频繁集的keys存储到新的频繁集
        for i in subset2:
            subset[i] = 0

        # 存储新的频繁集
        for i in subset2:
                for item_list in data:
                        # print("%s issubset %s ,%s"%(set(i),set(item_list),set(i).issubset(set(item_list))))
                        if set(i).issubset(set(item_list)):
                            # print(i)
                            subset[i] +=1
        # print('---------')
        # 删除不是值不为1的频繁集
        subset = {str(key):value for key,value in subset.items() if value !=1}

        # 更新到结果
        frequent.update(subset)

        # 新频繁集继续繁衍
        key_list = ()
        for i in subset:
            key_list+=eval(i)
        key_list = set(key_list)
        itemsubset = list(itertools.combinations(key_list,r=number+1))

        # 递归
        recursion_create(itemsubset)


    # 第一次输出元组类型的频繁集
    subset = {}
    for item_list in data:
        for item in item_list:
            if item not in subset.keys():
                subset[item] = 1
            else:
                subset[item] += 1

    subset = {key: value for key, value in subset.items() if value != 1}

    itemsubset = list(itertools.combinations(subset.keys(), r=2))

    frequent.update(subset)

    # 递归更新频繁集
    recursion_create(itemsubset)
    # print(subset)

    return frequent



def calcDegreeSet(frequent,degree_limit):
    degreeSet = {}
    for key, value in frequent.items():
        if '(' in key and ')' in key:
            subset = eval(key)

            # print(subset)
            for kid in subset:
                temp = list(subset)
                temp.remove(kid)
                temp.insert(0, kid)
                temp = tuple(temp)
                word_times = pinfanset[kid]

                degree =  round(value / word_times, 2)
                if degree > degree_limit:
                    degreeSet[temp] = degree

    return degreeSet



if __name__ == '__main__':
    data = load_dataset()
    print(data)
    pinfanset = calcFrequent(data)
    degreeSet = calcDegreeSet(pinfanset,0.1)
    print(degreeSet)
