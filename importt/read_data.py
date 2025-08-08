import pandas as pd
import numpy as np

class ReadData():
    """
    读取数据，并转为列表的标准形式
    """
    def __init__(self, args):
        self.args = args
    
    def run(self, X):
        """
        :return (pd.DataFrame), index为pd.Timestamp日期，第一列为因变量，第二列为因变量的预测值，其他列为辅助的气象自变量
        """
        # 取负荷数据

        df_power = pd.read_csv('data/processed/gd_newEnergy_data_0728.csv')
        # cols = [self.args.time_col, 'haifeng_True', 'haifeng_Pred']
        # cols = [self.args.time_col, 'pv_True', 'pv_Pred']
        # cols = [self.args.time_col, 'dpv_True', 'dpv_Pred']
        # cols = [self.args.time_col, 'lufeng_True', 'lufeng_Pred']
        cols = [self.args.time_col, 'newEnergy_True', 'newEnergy_Pred']
        
        # # # 光伏数据处理
        # pv_columns = [col for col in df_power.columns if 'pv' in col.lower()]
        # start_time = pd.to_datetime('06:30:00').time()
        # end_time = pd.to_datetime('19:00:00').time() 
        # df_power['Date'] = pd.to_datetime(df_power['Date'])      
        # # 创建时间掩码
        # time_mask = (df_power['Date'].dt.time <= start_time) | (df_power['Date'].dt.time >= end_time)
        # # 将符合条件的数据置为0
        # df_power.loc[time_mask, pv_columns] = 0
        # df_power['nnnewEnergy_True'] = df_power['haifeng_True'] + df_power['lufeng_True'] + df_power['pv_True'] + df_power['dpv_True']
        # df_power['nnnewEnergy_Pred'] = df_power['haifeng_Pred'] + df_power['lufeng_Pred'] + df_power['pv_Pred'] + df_power['dpv_Pred']
        # cols = [self.args.time_col, 'nnnewEnergy_True', 'nnnewEnergy_Pred']
        df_power = df_power[cols]
        
        # 功率数据根据装机情况归一化
        df_scaled = self.__scale_power(df_power, '风光')  #海上风电, 集中式光伏， 分布式光伏， 陆上风电，风光
        
        # 取日前预报气象数据
        temp = pd.read_csv('data/graph_a237_object_YHY003280_node_a960_run.csv')

        temp['time'] = pd.to_datetime(temp['time'])
        cond = (temp['time'] >= pd.Timestamp('2024.06.01')) & (temp['time'] <= pd.Timestamp('2025.09.01'))
        temp = temp[cond]
        temp = temp.set_index('time')
        
        # df_wind = self.__get_wind_speed_cube(temp)
        df_wind = self.__get_ssr(temp)
        df_wind = df_wind.reset_index().rename(columns={'time': self.args.time_col})
        
        DF = pd.merge(df_scaled, df_wind, on=self.args.time_col)
            
        # 设置time_col为index列    
        # return df_scaled.set_index(self.args.time_col)
   
        return DF.set_index(self.args.time_col)
    
    def predict(self):
        pass
    
    def __scale_power(self, df_power, mark='海上风电'):
        df = df_power.copy()
        
        df_capacity = pd.read_excel('data/capacity.xlsx')
        df_capacity["month"] = pd.to_datetime(df_capacity[self.args.time_col]).dt.to_period("M")
        
        df[self.args.time_col] = pd.to_datetime(df[self.args.time_col])
        df["month"] = df[self.args.time_col].dt.to_period("M")
        
        df_merged = pd.merge(df, df_capacity[['month', mark]], on="month", how="inner")
        count_list = [self.args.time_col]
        for col in list(df_power.columns)[1:]:
            df_merged[f"normalized_{col}"] = df_merged[col] / df_merged[mark]
            count_list.append(f"normalized_{col}")

        df_normalized = df_merged[count_list]
        return df_normalized
    
    def __get_wind_speed_cube(self, temp):
        
        
        wind_columns = [col for col in temp.columns if '100_' in col.lower()]
        temp = temp[wind_columns]
        
        df = pd.DataFrame(index=temp.index)
        
        station_df = pd.read_csv('data/gd_cmaStation.csv')
        my_list = station_df['station_id'].unique().tolist()
        my_list = [str(i)  for  i in my_list]
        for station in my_list:
            df[f'{station}_100m^3'] = np.sqrt( temp[f'u100_{station}']**2 + temp[f'v100_{station}']**2 ) ** 3 # 计算风速的三次方   
            
        return df   
    
    def __get_ssr(self, temp):
        ssr_columns = [col for col in temp.columns if 'ssr' in col.lower()]
        temp = temp[ssr_columns]
        
        return temp
          
        
        