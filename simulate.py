import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd

# # 固定参数
# mu = 0
# sigma = 1.0

# # 定义目标函数
# def objective(r1, a, b):
#     z = r1 / sigma
#     term1 = a * r1
#     term2 = b * (sigma * norm.pdf(z) - r1 * (1 - norm.cdf(z)))
#     return term1 + term2

# # 网格搜索范围
# a_values = np.linspace(0.1, 5, 50)
# b_values = np.linspace(0.1, 5, 50)

# # 存储结果
# results = np.zeros((len(a_values), len(b_values)))

# for i, a in enumerate(a_values):
#     for j, b in enumerate(b_values):
#         res = minimize_scalar(lambda r1: objective(r1, a, b), bounds=(0, 3*sigma), method='bounded')
#         results[i, j] = res.x

# # 转换为二维数组后绘制热力图
# A, B = np.meshgrid(b_values, a_values)

# plt.figure(figsize=(10, 8))
# contour = plt.contourf(B, A, results, levels=50, cmap='viridis')
# plt.colorbar(contour, label='Optimal r1')
# plt.xlabel('b (tail risk cost)')
# plt.ylabel('a (base reserve cost)')
# plt.title('Optimal Reserve Level r1 under different (a, b)')
# plt.grid(True)
# plt.savefig('aaa.png')
# plt.show()

# # 绘制图像
# plt.figure(figsize=(8, 6))
# plt.plot(b_values, optimal_r1s, label=f'$a = {a}$', marker='o', markersize=4)
# plt.xlabel('b (Extreme Risk Penalty)')
# plt.ylabel('Optimal r1')
# plt.title('Optimal Reserve Level r1 vs. Extreme Risk Penalty b')
# plt.grid(True)
# plt.legend()
# plt.savefig('bbb.png')
# plt.show()


# aa = [i+mu for i in optimal_r1s]
# plt.figure(figsize=(8, 6))
# plt.plot(b_values, 2*norm.cdf(aa)-1, label=f'$a = {a}$', marker='o', markersize=4)
# plt.xlabel('b (Extreme Risk Penalty)')
# plt.ylabel('Confidence Interval')
# plt.title('Optimal Reserve Level r1 vs. Extreme Risk Penalty b')
# plt.grid(True)
# plt.legend()
# plt.savefig('ccc.png')
# plt.show()


import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt



def returnPosNeg(mu, p, r1):
    """
    可以先按mu均值优化对称的r1，然后再做计算吗？ 
    --好像不行，因为两项成本，系数不一致，尤其在尾部的期望成本部分，高度非线性，所以"先对称优化再根据预测功率值做计算",
      与非对称优化不等价。要达到分开考虑正负备用的不同成本系数，还是得重新分开构造优化目标函数, 
      正如opt_func函数中的 func='LinearPos','LinearNeg'所示.
    
    :mu (均值, float)
    :p  (预测功率值， float)
    :r1 (基于正态均值对称求解的备用容量， float, 正数)
    
    :return pos 新能源专项正备用容量
    :       neg 新能源专项负备用容量
    """
    if p < mu:  # 概率模型调整值mu 大于新能源厂家预测功率 p
        if abs(p-mu) >= r1:
            pos = 0
            neg = r1 + mu - p
            
        else:
            pos = r1 - (mu - p)
            neg = r1 + (mu - p)
    else: 
        if abs(p-mu) >= r1:
            pos = r1 + (p - mu)
            neg = 0
            
        else:
            pos = r1 + (p - mu)
            neg = r1 - (p - mu)
            
    return pos, neg
            
            
        

# 参数设置
# 存储结果
optimal_r1s = []
def opt_func(mu, sigma, a, b, p=None, func='linear'):
    

    # 定义目标函数
    if func=='linear':
        def objective(r1, a, b, p):
            z = r1 / sigma
            term1 = a * r1
            term2 = b * (sigma * norm.pdf(z) - r1 * (1 - norm.cdf(z)))
            return term1 + term2

    elif func == 'quadratic':
        def objective(r1, a, b, p):
            z = r1 / sigma
            term1 = a * r1
            term2 = b * ((2*mu*sigma-r1*sigma) * norm.pdf(z) + (mu-r1)**2 - (sigma**2 - (r1-mu)**2) * norm.cdf(z))
            return term1 + term2
        
    elif func == 'LinearPos':
        def objective(r1, a, b, p):
            z = (p - r1 - mu) / sigma
            term1 = a * r1
            term2 = b * (sigma * norm.pdf(z) + z * sigma * (norm.cdf(z)))
            return term1 + term2
        
    elif func == 'LinearNeg':
        def objective(r1, a, b, p):
            z = (p + r1 - mu) / sigma
            term1 = a * r1
            term2 = b * (sigma * norm.pdf(z) - z * sigma * (1 - norm.cdf(z)))
            return term1 + term2

    res = minimize_scalar(lambda r1: objective(r1, a, b, p), bounds=(0, 3*sigma), method='bounded')
    return res

# a=1
# # b_values = np.linspace(0.1, 30, 100)  # 极端风险惩罚系数的变化范围
# b_values = [1, 3, 5, 10, 15, 20, 25, 30]
# # b_values = [15]
# model = 'LinearPos'

# DF = pd.read_csv('newEnergy_总量.csv', encoding='gbk')
# # 根据不同时刻点
# for b in b_values:
#     k = b/a
#     print('开始，b=', b)
#     optimal_resPos = []
#     optimal_confidencePos = []
#     optimal_securityPos = []
#     optimal_resNeg = []
#     optimal_confidenceNeg = []
#     optimal_securityNeg = []
    
#     optimal_costs_pred = []
#     optimal_costs_real = []
#     optimal_is_secu = []
#     for i, (mu, sigma) in enumerate(zip(DF['mu'], DF['sigma'])):
#         resPos = opt_func(mu, sigma, a, b, 'LinearPos')
#         resNeg = opt_func(mu, sigma, a, b, 'LinearNeg')
#         p = DF.loc[i, 'newEnergy_Pred']
#         true = DF.loc[i, 'newEnergy__True']
        
#         def calcu_conf(model = 'Neg', r1=1, p=1, mu=1, sigma=1):
#             if r1<=0.1:
#                 aa = 0
#             else:
#                 if p < 7895.3:  #mu > p
#                     if 'Pos' in model:
#                         aa = (r1 + (mu - p))/sigma
#                     else:
#                         aa = abs(r1 - (mu - p))/sigma
#                 else:
#                     if 'Pos' in model:
#                         if p-r1 > mu:
#                             buyd=1
#                         aa = abs(r1 - (p - mu))/sigma
#                     else:
#                         aa = (r1 + (p - mu))/sigma
#             conf = max(2*norm.cdf(aa)-1, 0)
#             return conf
        
#         # 正备用计算
#         optimal_resPos.append(resPos.x)

#         optimal_confidencePos.append(calcu_conf('Pos', resPos.x, p, mu, sigma))
#         if p - true <= resPos.x:
#             secu = 0
#         else:
#             secu = p - true - resPos.x
#         optimal_securityPos.append(secu)
#         # 负备用计算
#         optimal_resNeg.append(resNeg.x)

#         optimal_confidenceNeg.append(calcu_conf('Neg', resNeg.x, p, mu, sigma))
#         if true - p <= resNeg.x:
#             secu = 0
#         else:
#             secu = true - p - resNeg.x
#         optimal_securityNeg.append(secu)
        

#     DF[f'Linear_B={k}A_专项正备用容量'] = optimal_resPos
#     DF[f'Linear_B={k}A_专项正备用对应置信水平'] = optimal_confidencePos
#     DF[f'Linear_B={k}A_占用安全裕度正备用容量'] = optimal_securityPos
#     DF[f'Linear_B={k}A_专项负备用容量'] = optimal_resNeg
#     DF[f'Linear_B={k}A_专项负备用对应置信水平'] = optimal_confidenceNeg
#     DF[f'Linear_B={k}A_占用安全裕度负备用容量'] = optimal_securityNeg
#     print('结束，b=', b)
    
#     # if k % 3 == 0:
# DF.to_excel(f'data/out/newEnergy_总量_仿真结果quan.xlsx')
        

# 曲线图
dict_pSigma = {2500: 1582.5,
               7000: 1334.5,
               11000: 1792,
               18000: 2463,
               27000: 2245}
a=1
b_values = np.linspace(0.1, 40, 100)  
model = 'LinearPos'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"{model[-3:]} Reserves", fontsize=16, fontweight='bold')
for p, sigma in dict_pSigma.items():
    mu = 0.8071*p + 1523  # 对p的校正线性函数，当p < 7895.3时候，mu > p
    optimal_r1s = []
    for b in b_values:
        res = opt_func(mu, sigma, a, b, p, func=model)
        optimal_r1s.append(res.x)
    
    # 绘制图像
    ax1.plot(b_values, optimal_r1s, label=f'pre_value = {p} MW', marker='o', markersize=4)
    ax1.set_xlabel('b (Extreme Risk Penalty)')
    ax1.set_ylabel('Optimal Reserve (MW)')
    ax1.set_title(f'Optimal Reserve Level r_{model[-3:]} vs. Extreme Risk Penalty b')
    ax1.grid(True)
    ax1.legend()



    optimal_confs = []
    for r1 in optimal_r1s:
        if r1<=0.1:
            aa = 0
        else:
            if p < 7895.3:  #mu > p
                if 'Pos' in model:
                    aa = (r1 + (mu - p))/sigma
                else:
                    aa = abs(r1 - (mu - p))/sigma
            else:
                if 'Pos' in model:
                    aa = abs(r1 - (p - mu))/sigma
                else:
                    aa = (r1 + (p - mu))/sigma
        conf = max(2*norm.cdf(aa)-1, 0)
        optimal_confs.append(conf)
                
    ax2.plot(b_values, optimal_confs, label=f'pre_value = {p} MW', marker='o', markersize=4)
    ax2.set_xlabel('b (Extreme Risk Penalty)')
    ax2.set_ylabel('Confidence Interval')
    ax2.set_title(f'Optimal Reserve Level Confidence vs. Extreme Risk Penalty b')
    ax2.grid(True)
    ax2.legend()
    
plt.tight_layout()
plt.savefig(f'{model}.png')
