import math
import Remove as re
import pandas as pd
from scipy.optimize import fsolve

Df = pd.DataFrame({
    'x1': [42, 55],
    'x2': [33, 34],
    'x3': [66, 45],
    'x4': [34, 56],
    'x5': [22, 45]},
    index=['m1', 'm2']
)

def paixu(df):
    [h, l] = df.shape  # 获取输入的行列大小
    ex = []  # ex记录所有出现过的排序结果
    nd = []  # nd记录排序后的新数据
    for i in range(0, h):
        df1 = df.iloc[i:i + 1, 1:l ] #返回指定行的所有列
        df1.index = [1]
        df1 = df1.sort_values(by=1, axis=1)
        ex.append(df1.columns.values)
        nd.append(df1.iloc[0].tolist())
    return [ex, nd]

def xiabiao(data, complete_element):
    e_numbers = []  # 统计该样本特征组合子集的下标
    x_numbers = []  # 统计排序下标
    [Aa, nd] = paixu(data)  # 排序的特征
    A = re.zji(Aa)  # 特征组合子集
    Aa = [[item] for sublist in Aa for item in sublist]  # 切分开
    for a in A:
        e_numbers.append(complete_element.index(a))
    for s in Aa:
        x_numbers.append(complete_element.index(s))
    return [e_numbers, x_numbers, nd]


# 多项式求解
def equation(a, x):
    return a + 1 - (a * x[0] + 1) * (a * x[1] + 1) * (a * x[2] + 1) * (a * x[3] + 1) * (a * x[4] + 1) * (
            a * x[5] + 1) * (a * x[6] + 1) * (a * x[7] + 1) * (a * x[8] + 1)


def polynomialsolution(l, x):
    lamta = fsolve(equation, x0=l, args=x)
    return lamta


def derivation_ltog(lamta, g, x):
    lg = lamta ** 2 + lamta
    sm = 0.0
    for i in range(0, 9):
        sm = sm + (x[i] / (1 + x[i] * lamta))
    lg = lg / ((1 + g * lamta) * 1 - (lamta + 1) * sm)
    return lg


def derivation_atog(i, lamta, x_numbers, g, gi, x, e_numbers, useless_number):
    if i == 9 or e_numbers[i] in useless_number:
        ag = 0.0
    else:
        if i + 1 > 8:
            A = 0.0
        else:
            A = x[e_numbers[i + 1]]
        if x_numbers[i] == gi:
            ag = 1 + lamta * A + x[x_numbers[i]] * A * derivation_ltog(lamta, g, x) + \
                 (1 + lamta * x[x_numbers[i]]) * derivation_atog(i + 1, lamta, x_numbers, g, gi, x, e_numbers,
                                                                 useless_number)
        else:
            ag = (1 + lamta * x[x_numbers[i]]) * derivation_atog(i + 1, lamta, x_numbers, g, gi, x, e_numbers,
                                                                 useless_number) + \
                 x[x_numbers[i]] * A * derivation_ltog(lamta, g, x)
    # 对判断是数据支持就递归不是数据支持就变0
    return ag


def derivation_ctog(lamta, x, g, gi, useless_number, data, complete_element):  # x为当代测度数组，data为某一样本的各特征观测值，gi表示测度的下标
    cg = 0.0
    [e_numbers, x_numbers, new_data] = xiabiao(data,
                                               complete_element)  # e_numbers是所有的子集的下标 x_numbers是排序下标 new_data新排序下的观测值
    for i in range(0, 9):
        if i == 0:
            cg = derivation_atog(i, lamta, x_numbers, g, gi, x, e_numbers, useless_number) * new_data[0][x_numbers[i]]
        else:
            cg = cg + derivation_atog(i, lamta, x_numbers, g, gi, x, e_numbers, useless_number) * \
                 (new_data[0][x_numbers[i]] - new_data[0][x_numbers[i - 1]])
    return cg


# 计算choqute积分
def integral_cf(x, data, complete_element, array):
    [e_numbers, x_numbers, new_data] = xiabiao(data, complete_element)
    cf = 0.0
    for i in range(0, 9):
        if i == 8:
            cf = cf + array[x_numbers[i]][int(new_data[0][i] * 10) - 1] * x[e_numbers[i]]
        else:
            cf = cf + array[x_numbers[i]][int(new_data[0][i] * 10) - 1] * (x[e_numbers[i]] - x[e_numbers[i + 1]])
    return cf


# 计算不相似度量小于等于0表示分类正确
def di(x1, x2, data, complete_element, c, barrary, marrary):  # c表示类别 x1表示良性 x2表示恶性 data为输入的一个dataframe格式样本
    d = 0.0
    if c == 2:
        d = -1 * integral_cf(x1, data, complete_element, barrary) + \
            integral_cf(x2, data, complete_element, marrary)
    elif c == 4:
        d = -1 * integral_cf(x2, data, complete_element, marrary) + \
            integral_cf(x1, data, complete_element, barrary)
    else:
        print('类标签异常')
    return d


# 损失函数
def lossfunction(di, a):
    li = 0.0
    if di <= 0:
        li = 0.0
    elif di > 0:
        li = 2 * (1.0 / (1 + math.exp(-1 * a * di)) - 0.5)
    return li


# 针对一个测度的梯度
def grad_cost(alpha, c, lamta1, lamta2, x1, x2, g, gi, useless_number1, useless_number2, datas, complete_element,
              barrary, marrary):  # c表示测度类别，x1x2表示两类的测度
    grad = 0.0
    b_loss = 0.0
    m_loss = 0.0
    grouped = datas.groupby('Class')
    Benign_datas = grouped.get_group(2)  # 良性肿瘤数据
    Malignant_datas = grouped.get_group(4)  # 恶性肿瘤数据

    [h, l] = Benign_datas.shape
    for i in range(0, h):
        data = pd.DataFrame(Benign_datas.iloc[i:i + 1])
        d = di(x1, x2, data, complete_element, 2, barrary, marrary)
        li = lossfunction(d, alpha)
        if c == 2:
            b_loss = b_loss + li * (1 - li) * (-1) * derivation_ctog(lamta1, x1, g, gi, useless_number1, data,
                                                                     complete_element)
        elif c == 4:
            b_loss = b_loss + li * (1 - li) * derivation_ctog(lamta2, x2, g, gi, useless_number2, data,
                                                              complete_element)
        else:
            print('类标号错误')

    [h, l] = Malignant_datas.shape
    for i in range(0, h):
        data = pd.DataFrame(Malignant_datas.iloc[i:i + 1])
        d = di(x1, x2, data, complete_element, 4, barrary, marrary)
        li = lossfunction(d, alpha)
        if c == 4:
            m_loss = m_loss + li * (1 - li) * (-1) * derivation_ctog(lamta2, x2, g, gi, useless_number2, data,
                                                                     complete_element)
        elif c == 2:
            m_loss = m_loss + li * (1 - li) * derivation_ctog(lamta1, x1, g, gi, useless_number1, data,
                                                              complete_element)
        else:
            print('类标号错误')

    grad = b_loss + m_loss
    return grad


def correct(x1, x2, datas, complete_element, barrary, marrary):
    grouped = datas.groupby('Class')
    Benign_datas = grouped.get_group(2)  # 良性肿瘤数据
    Malignant_datas = grouped.get_group(4)  # 恶性肿瘤数据
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    [h1, l] = Benign_datas.shape
    for i in range(0, h1):
        data = pd.DataFrame(Benign_datas.iloc[i:i + 1])
        d = di(x1, x2, data, complete_element, 2, barrary, marrary)
        if d > 0:
            fp = fp + 1
        else:
            tn = tn + 1
    [h2, l] = Malignant_datas.shape
    for i in range(0, h2):
        data = pd.DataFrame(Malignant_datas.iloc[i:i + 1])
        d = di(x1, x2, data, complete_element, 4, barrary, marrary)
        if d <= 0:
            tp = tp + 1
        else:
            fn = fn + 1
    return [1.0 * tp / (tp+fp),  1.0*(tp+tn) / (h1+h2)]
