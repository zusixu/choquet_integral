import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import Remove as re
import TrainFunction as tr

data = pd.read_csv('data.csv')
grouped = data.groupby('Class')
Benign_data = grouped.get_group(2)  # 良性肿瘤数据
Malignant_data = grouped.get_group(4)  # 恶性肿瘤数据
# Benign_text_data = Benign_data.sample(frac=0.65, random_state=6)  # 良性测试集
# Malignant_text_data = Malignant_data.sample(frac=0.35, random_state=35)  # 恶性测试集
# text_data = Benign_text_data._append(Malignant_text_data)  # 测试集
Benign_array = np.loadtxt("benign_array.txt")
Malignant_array = np.loadtxt("malignant_array.txt")

[useless_element1, complete_element] = re.remove(Benign_data)  # 记录数据不支持变量集合以及全集
[useless_element2, complete_element] = re.remove(Malignant_data)
useless_number1 = []  # 记录数据不支持变量的下标
for element in complete_element:
    if element in useless_element1:
        useless_number1.append(complete_element.index(element))
useless_number2 = []  # 记录数据不支持变量的下标
for element in complete_element:
    if element in useless_element2:
        useless_number2.append(complete_element.index(element))

# 初始值
a = np.random.rand(2, 9)
scale0 = 1 / (a[0].sum() + 0.1)  # 避免分母为0
a[0] = a[0] * scale0
scale1 = 1 / (a[1].sum() + 0.1)  # 避免分母为0
a[1] = a[1] * scale1
x1 = np.zeros(511)  # 良性
for num in range(0, 9):
    x1[num] = a[0][num]
x1[-1] = 1
newx1 = x1
lamta1 = tr.polynomialsolution(0.5, x1[0:9])
x2 = np.zeros(511)
for num in range(0, 9):
    x2[num] = a[1][num]
x2[-1] = 1
newx2 = x2
lamta2 = tr.polynomialsolution(0.5, x2[0:9])


def measure(X, element, lamta):  # 迭代算测度
    i = complete_element.index(element)
    if 8 >= i >= 0:
        x = X[i]
    else:
        new_element = element[1:]
        l_element = element[0:1]
        x = measure(X, l_element, lamta) + measure(X, new_element, lamta) + \
            measure(X, new_element, lamta) * measure(X, l_element, lamta) * lamta
    return x


for element in complete_element:
    i = complete_element.index(element)
    x1[i] = measure(x1, element, lamta1)

for element in complete_element:
    i = complete_element.index(element)
    x2[i] = measure(x2, element, lamta2)


best_correct = 0.0
a = 0.1 # 初始学习率
cs = []  # 历代准确率
le = []
Cr = []
best_x1 = x1
best_x2 = x2

for i in range(0, 50):

    c = random.randint(1, 1000)
    b = random.randint(100, 200)
    Benign_text_data = Benign_data.sample(n=15, random_state=c)  # 良性测试集
    Malignant_text_data = Malignant_data.sample(n=15, random_state=b)  # 恶性测试集
    text_data = pd.concat([Benign_text_data, Malignant_text_data])  # 测试集
    for j in range(0, 9):
        if j in useless_number1:
            newx1[j] = 0.0
        else:
            grad = tr.grad_cost(0.5, 2, lamta1, lamta2, x1, x2, x1[j], j, useless_number1, useless_number2, text_data,
                                complete_element, Benign_array, Malignant_array)
            newx1[j] = x1[j] - a * grad

    for n in range(0, 9):
        if n in useless_number2:
            newx2[n] = 0.0
        else:
            grad = tr.grad_cost(0.5, 4, lamta1, lamta2, x1, x2, x2[n], n, useless_number1, useless_number2, text_data,
                                complete_element, Benign_array, Malignant_array)
            newx2[n] = x2[n] - a * grad
    x1 = newx1
    x2 = newx2
    [cr,  c] = tr.correct(x1, x2, data, complete_element, Benign_array, Malignant_array)
    print('第', i, '代准确率为', c, 'best:', best_correct)
    cs.append(c)  # 存储准确率
    le.append(i)
    Cr.append(cr)
    if c > best_correct:
        best_correct = c
        best_x1 = x1
        best_x2 = x2
    # if sum(x1[0:9]) >= 1:
    #     print('x1',x1[0:9])
    # if sum(x2[0:9]) >= 1:
    #     print('x2',x2[0:9])
    c0 = c
    lamta1 = tr.polynomialsolution(0.5, x1[0:9])
    lamta2 = tr.polynomialsolution(0.5, x2[0:9])
    for element in complete_element:
        el = complete_element.index(element)
        x1[el] = measure(x1, element, lamta1)
    for element in complete_element:
        el = complete_element.index(element)
        x2[el] = measure(x2, element, lamta2)

with open('result1.pkl', 'wb') as f1:
    pickle.dump(best_x1, f1)
    f1.close()

with open('result2.pkl', 'wb') as f2:
    pickle.dump(best_x2, f2)
    f2.close()

with open('cr0105.pkl', 'wb') as f3:
    pickle.dump(Cr, f3)
    f3.close()

with open('c0105.pkl', 'wb') as f4:
    pickle.dump(cs, f4)
    f4.close()