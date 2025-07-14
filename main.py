import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


from importt.read_data import ReadData
from preprocessor.feature_select import FeatureSelect
from deeplearning.heteroscedastic_gaussian_regressor import HeteroscedasticGaussianRegressor, train_two_stage

class Main():
    def __init__(self):
        self.args = self.__parse_args()
      
    
    def solver(self):
        # 数据读取
        df = ReadData(self.args).run(1)
        
        # 特征筛选

        df = FeatureSelect().fit(df)
        
        # 数据准备
        trainx_tensor, trainy_tensor, valx_tensor, valy_tensor = self.__data_to_tensor(df)
        df_capacity = pd.read_excel('data/capacity.xlsx')
        df_capacity["month"] = pd.to_datetime(df_capacity[self.args.time_col]).dt.to_period("M")
        capacity = df_capacity[df_capacity['month']=='2025-04']['风光'].values[0]


        # 模型搭建和求解
        model = HeteroscedasticGaussianRegressor(input_dim=trainx_tensor.shape[1])
        train_two_stage(model, trainx_tensor, trainy_tensor)
        
        # 可视化
        self.__post_visualize(valx_tensor, valy_tensor, model, capacity)
        
        
    def __parse_args(self):
        parser = argparse.ArgumentParser(description="超参数配置")

        # 添加超参数
        parser.add_argument('--fc_sdate', type=str, default='2025-04-01', help='预测开始日期')
        parser.add_argument('--fc_edate', type=str, default='2025-04-30', help='预测结束日期')
        
        parser.add_argument('--train_days', type=int, default=90, help='使用近若干天的数据进行模型训练')
        
        parser.add_argument('--time_col', type=str, default='Date', help='日期列字段')
        
        parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
        parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
        parser.add_argument('--model_name', type=str, default='resnet18', help='模型名称')
        parser.add_argument('--use_cuda', action='store_true', help='是否使用GPU')

        args = parser.parse_args()
        return args
    
    def __data_to_tensor(self, df):
        train_data_cond = (df.index < pd.Timestamp(self.args.fc_sdate)) & (df.index >= (pd.Timestamp(self.args.fc_sdate) - pd.Timedelta(days=self.args.train_days)))
        self.train_data = df[train_data_cond]
        self.train_data = self.train_data.dropna()
        val_data_cond = (df.index >= pd.Timestamp(self.args.fc_sdate)) & (df.index <= pd.Timestamp(self.args.fc_edate))
        self.val_data = df[val_data_cond]
        self.val_data = self.val_data.interpolate()
        x_np, y_np = self.train_data.iloc[:, 1:].values, self.train_data.iloc[:, 0].values
        valx_np, valy_np = self.val_data.iloc[:, 1:].values, self.val_data.iloc[:, 0].values
        self.scaler_x = StandardScaler()  # 注意这里没有对输出进行标准化，最开始根据每月的装机对功率进行01标幺化
        if x_np.shape[1]>1:
            x_scaled = self.scaler_x.fit_transform(x_np[:, 1:])
            valx_scaled = self.scaler_x.transform(valx_np[:, 1:])
            x_scaled = np.concatenate([x_np[:,[0]], x_scaled], axis=1)
            valx_scaled = np.concatenate([valx_np[:,[0]], valx_scaled], axis=1)
        else:
            x_scaled = x_np.copy()
            valx_scaled = valx_np.copy()
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        valx_tensor = torch.tensor(valx_scaled, dtype=torch.float32)
        valy_tensor = torch.tensor(valy_np, dtype=torch.float32)
        
        return x_tensor, y_tensor, valx_tensor, valy_tensor
    
    
    def __post_visualize(self, x_tensor, y_tensor, model, capacity=1):
        length = x_tensor.shape[0]
        model.eval()
        with torch.no_grad():
            mu_pred, sigma2_pred = model(x_tensor)
            mu_pred = mu_pred.numpy()
            std_pred = np.sqrt(sigma2_pred.numpy()) 

        plt.figure(figsize=(8, 5))


        # 返回原数量级, 绘图
        plt.plot(np.arange(1, length+1),  np.array(y_tensor).ravel() * capacity, 'r', label='实际功率')
        # plt.plot(np.range(1, length+1), mu_pred* capacity, 'b', label='调整后的预测功率')
        plt.plot(np.arange(1, length+1), self.val_data.iloc[:,1].ravel()* capacity, 'g', label='预测功率')
        plt.fill_between(np.arange(1, length+1),
                        mu_pred * capacity - 2 * std_pred * capacity,
                        mu_pred * capacity + 2 * std_pred * capacity,
                        color='orange', alpha=0.3, label='±2 Std Dev')
        plt.title("Two-Stage Heteroscedastic Gaussian Regression")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
        # 计算指标，self.val_data
        DF = self.val_data.copy()
        for conf in [80, 90, 95]:
            a = (100-conf)/200
            DF[f'{conf}%下界'] = mu_pred * capacity - norm.ppf((1-a)) * std_pred * capacity
            DF[f'{conf}%下界'] = DF[f'{conf}%下界'].clip(lower=0)
            DF[f'{conf}%上界'] = mu_pred * capacity + norm.ppf((1-a)) * std_pred * capacity
            DF[f'{conf}%上界'] = DF[f'{conf}%上界'].clip(upper=capacity)
            DF[f'{conf}%覆盖率'] = ((DF.iloc[:,0] * capacity >= DF[f'{conf}%下界'] ) & (DF.iloc[:,0] * capacity <= DF[f'{conf}%上界'])).astype(int)
            DF[f'{conf}%平均带宽'] = DF[f'{conf}%上界'] - DF[f'{conf}%下界']
        a='nnnewEnergy'
        DF[f'{a}_Pred'] = DF[f'normalized_{a}_Pred'] * capacity
        DF[f'{a}__True'] = DF[f'normalized_{a}_True'] * capacity
        DF['mu'] = mu_pred * capacity
        DF['sigma'] = std_pred * capacity
        
        # DF = DF.between_time('6:45', '18:30')
        DF.loc[len(DF)] = DF.mean(axis=0)
        
        
        # 计算分量预测指标，self.val_data
        DF = self.val_data.copy()
        hf_std_pred = pd.read_csv('haifeng.csv', encoding='gbk')['sigma']
        lf_std_pred = pd.read_csv('lufeng_0.csv', encoding='gbk')['sigma']
        pv_std_pred = pd.read_csv('pv_00.csv', encoding='gbk')['sigma']
        dpv_std_pred = pd.read_csv('dpv_00.csv', encoding='gbk')['sigma']
        std_pred = (hf_std_pred + lf_std_pred + pv_std_pred + dpv_std_pred)[0:-1]
        hf_mu = pd.read_csv('haifeng.csv', encoding='gbk')['mu']
        lf_mu = pd.read_csv('lufeng_0.csv', encoding='gbk')['mu']
        pv_mu = pd.read_csv('pv_00.csv', encoding='gbk')['mu']
        dpv_mu = pd.read_csv('dpv_00.csv', encoding='gbk')['mu']
        mu_pred = (hf_mu + lf_mu + pv_mu + dpv_mu)[0:-1]
        for conf in [80, 90, 95]:
            a = (100-conf)/200
            DF[f'{conf}%下界'] = (mu_pred  - norm.ppf((1-a)) * std_pred).values
            DF[f'{conf}%下界'] = DF[f'{conf}%下界'].clip(lower=0)
            DF[f'{conf}%上界'] = (mu_pred + norm.ppf((1-a)) * std_pred).values
            DF[f'{conf}%上界'] = DF[f'{conf}%上界'].clip(upper=capacity)
            DF[f'{conf}%覆盖率'] = ((DF.iloc[:,0] * capacity  >= DF[f'{conf}%下界'] ) & (DF.iloc[:,0] * capacity  <= DF[f'{conf}%上界'])).astype(int)
            DF[f'{conf}%平均带宽'] = DF[f'{conf}%上界'] - DF[f'{conf}%下界']
        a='nnnewEnergy'
        DF[f'{a}_Pred'] = DF[f'normalized_{a}_Pred'] * capacity
        DF[f'{a}__True'] = DF[f'normalized_{a}_True'] * capacity
        DF['mu'] = mu_pred 
        DF['sigma'] = std_pred
        
        # DF = DF.between_time('6:45', '18:30')
        DF.loc[len(DF)] = DF.mean(axis=0)
        
        
        
    
        
if __name__ == "__main__":
    Main().solver()       
        
        

    