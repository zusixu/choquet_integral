import numpy as np
import pandas as pd
import itertools

data = pd.read_csv('breast-cancer-wisconsin.data',
                   names=['id number', 'Clump Thickness', 'Uniformity of Cell Size',
                          'Uniformity of Cell Shape',
                          'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                          'Normal Nucleoli',
                          'Mitoses', 'Class'], sep=",", skiprows=1)

# 预处理
data = data[data['Single Epithelial Cell Size'] != '?']  # 删除无效数据
data = data[data['Bare Nuclei'] != '?']
data['Bare Nuclei'] = data['Bare Nuclei'].astype('int')  # 这列是字符串数据转换为int
grouped = data.groupby('Class')
Benign_data = grouped.get_group(2)  # 良性肿瘤数据
Malignant_data = grouped.get_group(4)  # 恶性肿瘤数据
Benign_array = np.zeros([9, 10])
Malignant_array = np.zeros([9, 10])
for i in range(0, 9):
    b_counts = Benign_data.iloc[:, i + 1:i + 2].value_counts()
    m_counts = Malignant_data.iloc[:, i + 1:i + 2].value_counts()
    for j in range(0,10):
        if j + 1 in b_counts.index and j + 1 in m_counts.index:
            Benign_array[i][j] = 1.0 * b_counts[j + 1] / (b_counts[j + 1] + m_counts[j + 1])
            Malignant_array[i][j] = 1.0 * m_counts[j + 1] / (b_counts[j + 1] + m_counts[j + 1])
        elif j + 1 not in b_counts.index and j + 1 in m_counts.index:
            Benign_array[i][j] = 0.0
            Malignant_array[i][j] = 0.0
        elif j + 1 not in b_counts.index:
            Benign_array[i][j] = 0.0
            Malignant_array[i][j] = 1.0
        else:
            Benign_array[i][j] = 1.0
            Malignant_array[i][j] = 0.0
np.savetxt("benign_array.txt", Benign_array)
np.savetxt("malignant_array.txt", Malignant_array)

data[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
      'Single Epithelial Cell Size',
      'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']] \
    = data[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
            'Single Epithelial Cell Size',
            'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']] / 10
data.to_csv('data.csv', index=False)
Df = pd.DataFrame({
    'x1': [42, 55],
    'x2': [33, 34],
    'x3': [66, 45],
    'x4': [34, 56]},
    index=['m1', 'm2']
)


def paixu(df):
    [h, l] = df.shape  # 获取输入的行列大小
    ex = []  # ex记录所有出现过的排序结果
    nd = []  # nd记录排序后的新数据
    for i in range(0, h):
        df1 = df.iloc[i:i + 1, 1:l-1 ]
        df1.index = [1]
        df1 = df1.sort_values(by=1, axis=1)
        ex.append(df1.columns.values)
        nd.append(df1.iloc[0].tolist())
    return [ex, nd]


def jihe(dt):  # 所有组合
    a = []
    for i in range(1, len(dt) + 1):
        for combination in itertools.combinations(dt, i):
            a.append(sorted(combination))  # 利用sorted保证统一排序
    return a


def zji(Ex):
    re = []
    for i in range(0, len(Ex)):
        for j in range(0, 9):
            cp = Ex[i].tolist()
            del cp[0:j]
            re.append(sorted(cp))
    return re


def remove(dt):
    [Ex, nd] = paixu(dt)
    [h, l] = dt.shape
    dt = dt.iloc[:, 1:l-1 ]
    Dt = jihe(dt.columns.values)  # 数据全集
    # a=[' Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    re = zji(Ex)
    # for i in range(0,len(re)):
    #     if len(re[i])==1:
    #         re[i]=''.join(re[i])
    nan = []  # 数据不支持变量
    for element in Dt:  # sorted(dt.columns.values)
        if element not in re:
            nan.append(element)
    return [nan, Dt]
