import pandas as pd
import numpy as np
import pickle
import Remove as re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_csv('data.csv')
grouped = data.groupby('Class')
Benign_data = grouped.get_group(2)
Malignant_data = grouped.get_group(4)
Benign_text_data = Benign_data.sample(n=45, random_state=65)  # 良性测试集
Malignant_text_data = Malignant_data.sample(n=45, random_state=68)
text_data = pd.concat([Benign_text_data, Malignant_text_data])
with open('result1.pkl', 'rb') as f1:
    result = pickle.load(f1)
[useless_element1, complete_element] = re.remove(Benign_data)


def xiabiao(data, complete_element):
    e_numbers = []  # 统计该样本特征组合子集的下标
    x_numbers = []  # 统计排序下标
    [Aa, nd] = re.paixu(data)  # 排序的特征
    A = re.zji(Aa)  # 特征组合子集
    Aa = [[item] for sublist in Aa for item in sublist]  # 切分开
    for a in A:
        e_numbers.append(complete_element.index(a))
    for s in Aa:
        x_numbers.append(complete_element.index(s))
    return [e_numbers, x_numbers, nd]


def integral_cf(x, data, complete_element):
    [e_numbers, x_numbers, new_data] = xiabiao(data, complete_element)
    cf = 0.0
    for i in range(0, 9):
        if i == 8:
            cf = cf + new_data[0][i] * x[e_numbers[i]]
        else:
            cf = cf + new_data[0][i] * (x[e_numbers[i]] - x[e_numbers[i + 1]])
    return cf


def accuracy(measure, complet_element, train_data):
    [h, l] = train_data.shape
    x = np.empty(h)
    y = np.empty(h)
    correct = 0
    for i in range(0, h):
        data = pd.DataFrame(train_data.iloc[i:i + 1])
        x[i] = integral_cf(measure, data, complet_element)
        y[i] = data['Class'].values
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(x.reshape(-1, 1), y)
    x_pre = lda.predict(x.reshape(-1, 1))
    for l in range(0, h):
        if x_pre[l] == y[l]:
            correct += 1
    return correct


class set:
    def __init__(self):
        self.datas = []
        self.pre_sort = 0

    def euclidean_distance(self, data):
        data_vectors = data.iloc[:, 1:10].values
        datas_vectors = self.datas.iloc[:, 1:10].values
        diff = data_vectors - datas_vectors
        distance = np.sqrt(np.sum(np.square(diff), axis=1))
        return distance

    def choquet_distance(self, data):
        self_datas = self.datas.reset_index(drop=True)
        data = data.reset_index(drop=True)
        diff = data.copy()
        diff.iloc[:, 1:10] = (data.iloc[:, 1:10].values - self.datas.iloc[:, 1:10].values) ** 2
        distance = integral_cf(result, diff, complete_element)
        return distance

    def classfication_c(self, data):
        [h, l] = data.shape
        di = np.empty(h)  # 记录距离
        la = np.empty(h)  # 记录类别
        for i in range(0, h):
            da = pd.DataFrame(data.iloc[i:i + 1])
            di[i] = self.choquet_distance(da)
            la[i] = da['Class'].values
        min_indices = np.argpartition(di, 13)[:13]
        corresponding_la = la[min_indices]
        # Count the occurrences of 2 and 4 in corresponding_la
        count_2 = np.count_nonzero(corresponding_la == 2)
        count_4 = np.count_nonzero(corresponding_la == 4)
        # Print the result
        if count_2 > count_4:
            self.pre_sort = 2
        else:
            self.pre_sort = 4

    def classfication_e(self, data):
        [h, l] = data.shape
        di = np.empty(h)  # 记录距离
        la = np.empty(h)  # 记录类别
        for i in range(0, h):
            da = pd.DataFrame(data.iloc[i:i + 1])
            di[i] = self.euclidean_distance(da)
            la[i] = da['Class'].values
        min_indices = np.argpartition(di, 13)[:13]
        corresponding_la = la[min_indices]
        # Count the occurrences of 2 and 4 in corresponding_la
        count_2 = np.count_nonzero(corresponding_la == 2)
        count_4 = np.count_nonzero(corresponding_la == 4)
        # Print the result
        if count_2 > count_4:
            self.pre_sort = 2
        else:
            self.pre_sort = 4


[num, l] = text_data.shape
sets_1 = [set() for _ in range(num)]
sets_2 = [set() for _ in range(num)]
i = 0
correct_1 = 0
for se1 in sets_1:
    das = pd.DataFrame(text_data.iloc[i:i + 1])
    se1.datas = das
    se1.classfication_e(data)
    if se1.pre_sort == se1.datas['Class'].values:
        correct_1 += 1
    i += 1
print('欧氏距离准确率：', correct_1 / num)
j = 0
correct_2 = 0
for se2 in sets_2:
    das = pd.DataFrame(text_data.iloc[j:j + 1])
    se2.datas = das
    se2.classfication_c(data)
    if se2.pre_sort == se2.datas['Class'].values:
        correct_2 += 1
    j += 1
print('模糊积分距离准确率：', correct_2 / num)


correct_3 = accuracy(result, complete_element, text_data)
print('模糊积分准确率：', correct_3 / num)