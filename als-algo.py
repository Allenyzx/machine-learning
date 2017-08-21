import numpy as np
from math import pow
from numpy import nonzero

def alternating_least_squares(R,k=3,lamda=0.05,steps=100):

    # 生成矩阵R_(u,i)
    R = np.mat(R)

    # 行，列
    user,items = R.shape

    # 随机生成用户矩阵U_(u,k)
    Us = np.random.normal(size=(user,k))

    # 单位矩阵
    I = np.eye(k)

    # 迭代
    for step in range(steps):

        It =  np.dot(np.dot(R.T ,Us),np.linalg.inv(np.dot(Us.T,Us)+lamda*np.count_nonzero(R)*I) )

        e = 0
        for r in range(len(R)):
            for c in range(len(R[r])):
                e +=pow ((R[r,c] - np.dot(Us[r,:],It[c,:].T)[0,0]),2)

        Us = np.dot(np.dot(R ,It),np.linalg.inv(np.dot(It.T,It)+lamda*np.count_nonzero(R)*I) )
        print(e)

    print(np.dot(Us,It.T))


if __name__ == '__main__':
    R = [
        [1,2,],
        [1,0,],
        [1,2,],
        [3,4,]
    ]
    alternating_least_squares(R,k=10)
