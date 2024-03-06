import TrainFunction as tr
import pickle
import pandas as pd
import Remove as re
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv('breast-cancer-wisconsin.data',
#                    names=['id number', 'Clump Thickness', 'Uniformity of Cell Size',
#                           'Uniformity of Cell Shape',
#                           'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
#                           'Normal Nucleoli',
#                           'Mitoses', 'Class'], sep=",", skiprows=1)
#
# # 预处理
# data = data[data['Single Epithelial Cell Size'] != '?']  # 删除无效数据
# data = data[data['Bare Nuclei'] != '?']
# data['Bare Nuclei'] = data['Bare Nuclei'].astype('int')  # 这列是字符串数据转换为int
# data[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
#       'Single Epithelial Cell Size',
#       'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']] \
#     = data[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
#             'Single Epithelial Cell Size',
#             'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']] / 10
#
# [useless_element1, complete_element] = re.remove(data)
# Benign_array = np.loadtxt("benign_array.txt")
# Malignant_array = np.loadtxt("malignant_array.txt")
# with open('result1.pkl', 'rb') as f1:
#     result1 = pickle.load(f1)
#
# with open('result2.pkl', 'rb') as f2:
#     result2 = pickle.load(f2)
#
# c0 = tr.correct(result1, result2, data, complete_element,Benign_array,Malignant_array)
# print(c0)
# with open('cr0501.pkl', 'rb') as f1:
#     cr1 = pickle.load(f1)
# with open('cr0505.pkl', 'rb') as f3:
#     cr2 = pickle.load(f3)
# with open('cr0105.pkl', 'rb') as f4:
#     cr3 = pickle.load(f4)
# with open('time.pkl', 'rb') as f2:
#     tm = pickle.load(f2)
# plt.plot(tm, cr1, label='Line 1')
# plt.plot(tm, cr2, label='Line 2')
# plt.plot(tm, cr3, label='Line 3')
# plt.annotate('step=0.5 a=0.1', xy=(20, 0.946), xytext=(20, 0.962),
#              arrowprops=dict(arrowstyle='->'))
# plt.annotate('step=0.5 a=0.5', xy=(14, 0.934), xytext=(0.04, 0.962),
#              arrowprops=dict(arrowstyle='->'))
# plt.annotate('step=0.1 a=0.5', xy=(32, 0.96), xytext=(35, 0.962),
#              arrowprops=dict(arrowstyle='->'))
# plt.xlabel("times")
# plt.ylabel("sensitivity")
# plt.show()

with open('result1.pkl', 'rb') as f1:
    result1 = pickle.load(f1)

with open('result2.pkl', 'rb') as f2:
    result2 = pickle.load(f2)

with open('my_filer2.txt', 'w') as f:
    for item in result2:
        f.write("%s\n" % item)