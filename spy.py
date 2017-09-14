# Python3
# author:allenyzx
import os,sys
import pandas as pd
import numpy as np
import warnings
from collections import Counter
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
pd.set_option('display.width',2000)
pd.set_option('display.max_rows',2000)
data_path = os.path.dirname(__file__) + '/datasets/'



def load_datasets():
    # 读取数据集
    df = pd.read_csv(open(data_path + 'breast-cancer-wisconsin.data'), engine='python', index_col=False)
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'label']
    df['label'][np.where(df['label'].values == 2)[0]] = 1
    df['label'][np.where(df['label'].values == 4)[0]] = 0
    df = df.replace('?', 0).astype('int')

    # 1为正样本、0为未标注样本
    pudatasets = pd.DataFrame()
    pudatasets = pudatasets.append(df[df['label'] == 1][:50])
    pudatasets = pudatasets.append(df[df['label'] == 0])
    mixture = df[50:100]
    mixture['label'] = 0
    pudatasets = pudatasets.append(mixture)


    pudatasets = pudatasets.sample(len(pudatasets)).reset_index(drop=True)

    return pudatasets




class Spy:


    def __init__(self,psignal,usignal,splitpercent=0.15,step=10):
        """
        :param psignal: 正样本标注符号 
        :param usignal: 无标注样本标注符号
        :param splitpercent: 切割spy_data的占比
        :param step: 迭代步数
        """
        self.psignal = psignal
        self.usignal = usignal
        self.splitpercent = splitpercent
        self.step = step


    def spy_data(self,X,y):

        if type(X) is type(np.zeros(0)):
            X = pd.DataFrame(X)
        elif type(X) is type(pd.DataFrame()):
            pass
        else:
            raise ValueError('Structures is not np.array or pd.DataFrame')

        po_data = X.ix[np.where(y==self.psignal)[0],:]
        sample_num = int(self.splitpercent *len(po_data))

        s_data =  po_data.sample(sample_num)

        return s_data,s_data.index.values

    # step1:计算RN
    def _calcRN(self,X,y):
        s_data,s_index_list = self.spy_data(X,y)

        ps_data = self.po_data.copy()
        for i in s_index_list:
            ps_data.drop(i,axis=0,inplace=True)

        us_data = self.ul_data.append(s_data)

        pus_X = ps_data.append(us_data)
        pus_y = [self.psignal for _ in range(len(ps_data))] + [self.usignal for _ in range(len(us_data))]

        step = 0
        threshold = 0.
        rnIndexArray = []

        # 最大化取得RN
        while step < self.step:

            gmm = GMM(n_components=2)
            gmm.fit(pus_X,pus_y)
            pred = gmm.predict(s_data)

            po_num = 0
            for i in pred:
                if i == self.psignal:
                    po_num+=1

            every_t = po_num/len(pred)

            # 如果阈值为0，则跳过
            if every_t != 0:
                threshold += every_t

                us_data = us_data.reset_index(drop=True)
                u_pred = gmm.predict_proba(us_data)[:,0]

                rnIndexArray.extend([u_index for u_index in range(len(u_pred)) if u_pred[u_index] < every_t])


            if step == self.step - 1 and len(rnIndexArray) == 0:
                self.step +=1

            step += 1

        countsDict = Counter(rnIndexArray)
        maxIndex = max(countsDict.values())
        rnArray = []
        for key,value in countsDict.items():
            if value == maxIndex:
                rnArray.append(us_data.ix[key,:].tolist())


        self.t = threshold/self.step

        return np.asarray(rnArray)

    # step2:得到RN和正样本进行构建模型
    def classifier(self,X,y):
        step = 0
        predIndex = []

        # 尽量采集预测得到的正样本
        while step < self.step:
            rf = RandomForestClassifier(n_estimators=50)
            rf.fit(X,y)
            pred = rf.predict(self.ul_data)
            predIndex.extend([ _ for _ in range(len(pred)) if pred[_] ==self.psignal])
            step +=1

        self.ul_data = self.ul_data.reset_index(drop=True)

        predArray = np.asarray([self.ul_data.ix[i,:] for i in set(predIndex)])

        return predArray



    def predict(self,X,y):
        self.po_data = X.ix[np.where(y==self.psignal)[0],:]
        self.ul_data = X.ix[np.where(y==self.usignal)[0],:]
        rn = self._calcRN(X,y)

        po_data = X.ix[np.where(y==self.psignal)[0],:].as_matrix()

        X = np.concatenate((rn,po_data),axis=0)
        y = [self.usignal for _ in range(len(rn))]+[self.psignal for _ in range(len(po_data))]

        pred = self.classifier(X,y)

        return pred

    @property
    def get_threshold(self):
        return self.t





if __name__ == '__main__':


    data = load_datasets()

    spy = Spy(psignal=1,usignal=0)
    pred = spy.predict(X=data.ix[:,:-1],y=data.ix[:,-1])
    print(pred)
    print(spy.get_threshold)
