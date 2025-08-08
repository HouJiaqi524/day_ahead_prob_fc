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



def plot_interval(a, df, mark):
    plt.figure(figsize=(12, 5))
    plt.fill_between(df.index, df['95%下界'], df['95%上界'], 
            alpha=0.5, color='purple', label='95% 置信区间')
    plt.fill_between(df.index, df['90%下界'], df['90%上界'], 
                    alpha=0.7, color='purple', label='90% 置信区间')
    plt.fill_between(df.index, df['80%下界'], df['80%上界'], 
                    alpha=0.9, color='purple', label='80% 置信区间')

    plt.plot(df.index, df[f'{a}_Pred'],alpha=0.9, color='orange', linewidth=2, label=f'{a}_Pred')
    plt.plot(df.index, df[f'{a}__True'], alpha=0.9,color='red', linewidth=2, label=f'{a}_True')
    plt.legend()
    plt.savefig(f'{a}_{mark}',dpi=600)
    plt.show()
class Main():
    def __init__(self):
        self.args = self.__parse_args()
      
    
    def solver(self):
        # 数据读取
        df = ReadData(self.args).run(1)
        
        # 特征筛选

        df = FeatureSelect().fit(df)
        
        # df['distance_from_12'] = abs(df.index.hour * 60 + df.index.minute - 12 * 60) / 15
        
        self.a='newEnergy'  #haifeng, pv， dpv, lufeng, newEnergy
        self.m = '06'
        cap_name= '风光'  #海上风电, 集中式光伏 ,  分布式光伏， 陆上风电，风光
        
        # 数据准备
        trainx_tensor, trainy_tensor, valx_tensor, valy_tensor = self.__data_to_tensor(df, mark='day')
        df_capacity = pd.read_excel('data/capacity.xlsx')
        df_capacity["month"] = pd.to_datetime(df_capacity[self.args.time_col]).dt.to_period("M")
        capacity = df_capacity[df_capacity['month']==f'2025-{self.m}'][f'{cap_name}'].values[0]
        # 模型搭建和求解
        model = HeteroscedasticGaussianRegressor(input_dim=trainx_tensor.shape[1])
        train_two_stage(model, trainx_tensor, trainy_tensor, stage1_epochs=7000)
        # 计算
        DF = self.__post_visualize(valx_tensor, valy_tensor, model, capacity)
        
        if 'pv' in self.a:
            pass
        else:
            # 数据准备
            # if self.a == 'newEnergy':
            #     cap_name= '风电' 
            trainx_tensor, trainy_tensor, valx_tensor, valy_tensor = self.__data_to_tensor(df, mark='night')
            df_capacity = pd.read_excel('data/capacity.xlsx')
            df_capacity["month"] = pd.to_datetime(df_capacity[self.args.time_col]).dt.to_period("M")
            capacity = df_capacity[df_capacity['month']==f'2025-{self.m}'][cap_name].values[0]
            # 模型搭建和求解
            model = HeteroscedasticGaussianRegressor(input_dim=trainx_tensor.shape[1])
            train_two_stage(model, trainx_tensor, trainy_tensor, stage1_epochs=6000)
            # 计算
            DF1 = self.__post_visualize(valx_tensor, valy_tensor, model, capacity)
        
        DF2 = DF.copy()
        DF = DF.between_time('6:00', '18:00')
        DF2['1'] = 1
        DF3 = pd.merge(DF2[['1']], DF, left_index=True, right_index=True, how='left')
        cr_columns = DF3.columns[DF3.columns.str.contains('覆盖率', case=False)].tolist()
        DF3[cr_columns] = DF3[cr_columns].replace(np.nan, 1)
        DF3 = DF3.replace(np.nan, 0)
        
        if 'pv' in self.a:
            plot_interval(self.a,DF3, mark=self.m+'aft')   
            # DF1 = DF.between_time('6:45', '18:30')
            DF3.loc[len(DF3)] = DF3.mean(axis=0)
            DF3.to_csv(f'{self.a}_{self.m}_aft.csv', encoding='gbk')           
        else:
            DF1 = DF1[~DF1.index.isin(DF.index)]
            DF= pd.concat([DF,DF1], axis=0)
            DF = DF.replace(np.nan, 0)
            DF = DF.sort_index()
            # 可视化
            plot_interval(self.a,DF, mark=self.m+'aft')   
            # DF1 = DF.between_time('6:45', '18:30')
            DF.loc[len(DF)] = DF.mean(axis=0)
            DF.to_csv(f'{self.a}_{self.m}_aft.csv', encoding='gbk')
        
        # 三峰两谷统计
        DF.index = pd.to_datetime(DF.index)
        DF['hour'] = DF.index.hour * 4 + DF.index.minute / 15
        DF_11= DF.groupby('hour').mean()
        DF_11.to_csv(f'{self.a}_{self.m}_aft_三峰两谷.csv', encoding='gbk')
        # 分量预测加总
        # 计算分量预测指标，self.val_data
        DF = self.val_data.copy()
        hf_std_pred = pd.read_csv(f'haifeng_{self.m}_aft.csv', encoding='gbk')['sigma']
        lf_std_pred = pd.read_csv(f'lufeng_{self.m}_aft.csv', encoding='gbk')['sigma']
        pv_std_pred = pd.read_csv(f'pv_{self.m}_aft.csv', encoding='gbk')['sigma']
        dpv_std_pred = pd.read_csv(f'dpv_{self.m}_aft.csv', encoding='gbk')['sigma']
        std_pred = (hf_std_pred + lf_std_pred + pv_std_pred + dpv_std_pred)[0:-1]
        hf_mu = pd.read_csv(f'haifeng_{self.m}_aft.csv', encoding='gbk')['mu']
        lf_mu = pd.read_csv(f'lufeng_{self.m}_aft.csv', encoding='gbk')['mu']
        pv_mu = pd.read_csv(f'pv_{self.m}_aft.csv', encoding='gbk')['mu']
        dpv_mu = pd.read_csv(f'dpv_{self.m}_aft.csv', encoding='gbk')['mu']
        mu_pred = (hf_mu + lf_mu + pv_mu + dpv_mu)[0:-1]
        for conf in [80, 90, 95]:
            aa = (100-conf)/200
            DF[f'{conf}%下界'] = (mu_pred  - norm.ppf((1-aa)) * std_pred).values
            DF[f'{conf}%下界'] = DF[f'{conf}%下界'].clip(lower=0)
            DF[f'{conf}%上界'] = (mu_pred + norm.ppf((1-aa)) * std_pred).values
            DF[f'{conf}%上界'] = DF[f'{conf}%上界'].clip(upper=capacity)
            DF[f'{conf}%覆盖率'] = ((DF.iloc[:,0] * capacity  >= DF[f'{conf}%下界'] ) & (DF.iloc[:,0] * capacity  <= DF[f'{conf}%上界'])).astype(int)
            DF[f'{conf}%平均带宽'] = DF[f'{conf}%上界'] - DF[f'{conf}%下界']
        DF[f'{self.a}_Pred'] = DF[f'normalized_{self.a}_Pred'] * capacity
        DF[f'{self.a}__True'] = DF[f'normalized_{self.a}_True'] * capacity
        DF['mu'] = mu_pred 
        DF['sigma'] = std_pred
        
        plot_interval(self.a, DF, mark=self.m+'加和aft')   
        # DF1 = DF.between_time('6:45', '18:30')
        DF.loc[len(DF)] = DF.mean(axis=0)
        DF.to_csv(f'{self.a}_{self.m}_加和aft.csv', encoding='gbk')
        a=1
        
        
    def __parse_args(self):
        parser = argparse.ArgumentParser(description="超参数配置")

        # 添加超参数
        parser.add_argument('--fc_sdate', type=str, default='2025-06-01', help='预测开始日期')
        parser.add_argument('--fc_edate', type=str, default='2025-06-30', help='预测结束日期')
        
        parser.add_argument('--train_days', type=int, default=90, help='使用近若干天的数据进行模型训练')
        
        parser.add_argument('--time_col', type=str, default='Date', help='日期列字段')
        
        parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
        parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
        parser.add_argument('--model_name', type=str, default='resnet18', help='模型名称')
        parser.add_argument('--use_cuda', action='store_true', help='是否使用GPU')

        args = parser.parse_args()
        return args
    
    def __data_to_tensor(self, df, mark='normal'):
        train_data_cond = (df.index < pd.Timestamp(self.args.fc_sdate)) & (df.index >= (pd.Timestamp(self.args.fc_sdate) - pd.Timedelta(days=self.args.train_days)))
        self.train_data = df[train_data_cond]
        self.train_data = self.train_data.dropna()
        self.train_data_day = self.train_data.between_time('6:00', '18:00')
        self.train_data_night = self.train_data[~self.train_data.index.isin(self.train_data_day.index)]
        
        val_data_cond = (df.index >= pd.Timestamp(self.args.fc_sdate)) & (df.index <= pd.Timestamp(self.args.fc_edate))
        self.val_data = df[val_data_cond]
        self.val_data = self.val_data.interpolate()
        x_np_all, y_np_all = self.train_data.iloc[:, 1:].values, self.train_data.iloc[:, 0].values
        x_np_day, y_np_day = self.train_data_day.iloc[:, 1:].values, self.train_data_day.iloc[:, 0].values
        x_np_night, y_np_night = self.train_data_night.iloc[:, 1:].values, self.train_data_night.iloc[:, 0].values
        valx_np, valy_np = self.val_data.iloc[:, 1:].values, self.val_data.iloc[:, 0].values
        self.scaler_x = StandardScaler()  # 注意这里没有对输出进行标准化，最开始根据每月的装机对功率进行01标幺化
        if mark=='normal':
            x_np = x_np_all
            y_np = y_np_all
        elif mark=='day':
            x_np = x_np_day
            y_np = y_np_day
        elif mark=='night':
            x_np = x_np_night
            y_np = y_np_night
         
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
        a=self.a  #newEnergy
        m = self.m
        DF[f'{a}_Pred'] = DF[f'normalized_{a}_Pred'] * capacity
        DF[f'{a}__True'] = DF[f'normalized_{a}_True'] * capacity
        DF['mu'] = mu_pred * capacity
        DF['sigma'] = std_pred * capacity
        
        return DF
        
        
        
        
        

        
        
        
    
        
if __name__ == "__main__":
    Main().solver()       
        
        

    