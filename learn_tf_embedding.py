import pandas as pd
import numpy as np
import tensorflow as tf
import os,sys
pd.set_option('display.width',2000)
project_path = os.path.dirname(__file__)
import pandas as pd

def load_data():

    from sklearn.datasets import load_boston

    iris = load_boston()
    feature = iris.data
    label = iris.target
    feature_name = iris.feature_names
    # print(feature)
    data = pd.DataFrame(feature)
    data.columns = feature_name
    data['target'] = label

    data['sex'] = '1'

    data['sex'][data['ZN']>5] = 'male'
    data['sex'][data['ZN']<=5] = 'female'
    from sklearn.preprocessing import MinMaxScaler
    data['target'][data['target']<20] = 0
    data['target'][data['target']>=20] = 1

    return data


def input_fn(df):
  #"""这个函数的主要作用就是把输入数据转换成tensor，即向量型"""
  #"""Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  # 连续型变量
  # continuous_cols = {k: tf.constant(df[k].values) for k in ['sex']}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  # 类别型变量
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in ['sex']}
  # Merges the two dictionaries into one.
  # 保存到feature_cols(dict)
  feature_cols = dict(categorical_cols)
  # feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df['target'].values)
  # Returns the feature columns and the label.
  return feature_cols, label


if __name__ == '__main__':

    data = load_data()
    print(data)

    array = np.array([[1000, 0,2],[0,2,0],[3,3,3]])
    tfArray = tf.Variable(array, dtype=tf.float32)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        idx = tf.SparseTensor(indices=[[0,0], [1,0],[2,0]], values=[1,2,2], dense_shape=(2,2))

        embedding = tf.nn.embedding_lookup_sparse(tfArray,sp_ids=idx,sp_weights=None,combiner='sum')

        print(sess.run(embedding))


