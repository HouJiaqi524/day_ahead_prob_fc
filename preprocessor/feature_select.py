import pandas as pd
import numpy as np

class FeatureSelect():
    def __init__(self):
        pass
    
    def fit(self, x_, y=None):
        """
        筛选合适的自变量，用于后续建模。原则为，找到与因变量线性相关程度最高的一组自变量，同时减少自变量的耦合性
        
        """
        x = x_.copy()
        
        # 获取相关系数矩阵
        corr_matrix = x.corr()
        print("相关系数矩阵：")
        print(corr_matrix)
        
        target = x.columns[0]  # 因变量名，即第一列 
        self.corr_with_target = corr_matrix[target].drop(target).sort_values(ascending=False) # 取出与因变量的相关系数，并排序
        print("\n与因变量的相关系数（从高到低）：")
        print(self.corr_with_target)
        
        # 取排名前40的变量
        XX = x[list(self.corr_with_target[0:4].index)]
        selected_vars = self.remove_high_corr_vars(XX, threshold=0.97) # 获取要保留的自变量
        print("\n筛选后的自变量列表：")
        print(selected_vars)
        # selected_cols = [target] + selected_vars
        selected_cols = [target,  x.columns[1]]
        return x[selected_cols]
            
    def remove_high_corr_vars(self, df, threshold=0.9):
        """
        去除与其他变量高度相关的变量
        :param df: 自变量部分的 DataFrame
        :param threshold: 相关系数阈值
        :return: 被保留的变量列表
        """
        corr = df.corr()
        cols = list(corr.columns)
        to_remove = set()

        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if abs(corr.iloc[i, j]) > threshold:
                    # 保留与因变量更相关的那个
                    if abs(self.corr_with_target[cols[i]]) >= abs(self.corr_with_target[cols[j]]):
                        to_remove.add(cols[j])
                    else:
                        to_remove.add(cols[i])

        return [col for col in cols if col not in to_remove]
        
    
    def transform(self, x):
        pass