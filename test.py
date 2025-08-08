import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot1(DF, date_index='Date', mark='test'):
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for col in DF.columns[1:]:
        if '风电' in col or '光伏' in col or '负荷' in col or 'load' in col:
            secondary_y = False
        else:
            secondary_y = True
        fig.add_trace(
            go.Scatter(
                name=col,
                x=DF[date_index],
                y=DF[col],
            ),
            secondary_y=secondary_y,
        )
    fig.update_yaxes(title_text="<b>负荷</b> yaxis title", secondary_y=False)
    fig.update_yaxes(title_text="<b>气象</b> yaxis title", secondary_y=True)

    fig.write_html(f"{mark}.html")

g_sIndexCol = 'Date'
g_sPowerCol = 'True'
g_sPowerFcCol = 'Pred'

# 打印广东所有气象站点
station_df = pd.read_csv('data/gd_cmaStation.csv')
my_list = station_df['station_id'].unique().tolist()
my_list = [str(i)  for  i in my_list]
print(my_list)

from datetime import datetime, timedelta
def period_to_datetime(row, time_mark='PDATE'):
    # 将日期字符串转换为datetime对象
    base_date = pd.to_datetime(row[time_mark])
    # PERIODID 1 对应 00:00, 2对应00:15, 以此类推
    minutes_offset = (row['PERIODID'] - 1) * 15
    return base_date + timedelta(minutes=minutes_offset)

# 数据拼接,POWER是预测值，后面那个是实际值
####中调风电(海风)
df1 = pd.read_excel('data/20250520中调风电.xls')
df2 = pd.read_excel('data/装机及新能源/中调风电2506.xls')
df3 = pd.read_excel('data/新能源出力及装机20250721/中调风电实际67.xls').rename(columns={'F_CHANGE_LTODD(RECORDTIME)': 'PDATE', 'REALVALUE':'METVALUE'})
df4 = pd.read_excel('data/新能源出力及装机20250721/中调风电计划67.xlsx').rename(columns={'F_CHANGE_LTODD(RECORDTIME)': 'PDATE', 'POWERAS': 'POWER'})
df4['PDATE'] = pd.to_datetime(df4['PDATE'])
df3['PDATE'] = pd.to_datetime(df3['PDATE'])
df4['PDATE'] = df4.apply(period_to_datetime, axis=1, time_mark='PDATE')
df3.columns = ['PDATE', 'METVALUE']
df5 = pd.merge(df4, df3, on='PDATE')
df5.pop('PERIODID')
df = pd.concat([df1, df2, df5], axis=0)
df['PDATE'] = pd.to_datetime(df['PDATE'])
df = df.rename(columns={'PDATE':g_sIndexCol, 'POWER': f'haifeng_{g_sPowerFcCol}', 'METVALUE': f'haifeng_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_haifeng = df.drop_duplicates(g_sIndexCol)


### 地调风电（陆风）
df1 = pd.read_excel('data/装机及新能源/地调风电2506.xls')
df2 = pd.read_excel('data/20250520地调风电.xls')
df3 = pd.read_excel('data/新能源出力及装机20250721/地调风电实际67.xlsx').rename(columns={'F_CHANGE_LTODD(RECORDTIME)': 'PDATE', 'REALVALUE':'METVALUE'})
df4 = pd.read_excel('data/新能源出力及装机20250721/地调风电计划67.xlsx').rename(columns={'F_CHANGE_LTODD(RECORDTIME)': 'PDATE', 'POWERAS': 'POWER'})
df3.columns = ['PDATE', 'METVALUE']
df4['PDATE'] = pd.to_datetime(df4['PDATE'])
df3['PDATE'] = pd.to_datetime(df3['PDATE'])
df4['PDATE'] = df4.apply(period_to_datetime, axis=1, time_mark='PDATE')
df5 = pd.merge(df4, df3, on='PDATE')
df5.pop('PERIODID')
df = pd.concat([df1, df2, df5], axis=0)
df['PDATE'] = pd.to_datetime(df['PDATE'])
df = df.rename(columns={'PDATE':g_sIndexCol, 'POWER': f'lufeng_{g_sPowerFcCol}', 'METVALUE': f'lufeng_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_lufeng = df.drop_duplicates(g_sIndexCol)


### 集中式光伏
df = pd.read_excel('data/装机及新能源/中调集中式光伏2506.xlsx')
df3 = pd.read_excel('data/新能源出力及装机20250721/集中式光伏实际67.xlsx').rename(columns={'F_CHANGE_LTODD(RECORDTIME)': '日期', 'REALVALUE':'实际出力'})
df4 = pd.read_excel('data/新能源出力及装机20250721/集中式光伏计划67.xlsx').rename(columns={'PDATE': '日期', 'POWER': '计划出力'})
df4['日期'] = pd.to_datetime(df4['日期'])
df4['日期'] = df4.apply(period_to_datetime, axis=1, time_mark='日期')
df3.columns = ['日期', '实际出力']
df5 = pd.merge(df4, df3, on='日期')
df5.pop('PERIODID')
df = pd.concat([df, df5], axis=0)
df['日期'] = pd.to_datetime(df['日期'])
df = df.rename(columns={'日期':g_sIndexCol, '计划出力': f'pv_{g_sPowerFcCol}', '实际出力': f'pv_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_pv = df.drop_duplicates(g_sIndexCol)


### 分布式光伏
df1 = pd.read_excel('data/装机及新能源/分布式光伏2506.xls')
df2 = pd.read_excel('data/20250520分布式光伏.xls')
df3 = pd.read_excel('data/新能源出力及装机20250721/分布式光伏实际67.xlsx')
df4 = pd.read_excel('data/新能源出力及装机20250721/分布式光伏计划67.xlsx')
df4['PDATE'] = pd.to_datetime(df4['PDATE'])
df4['PDATE'] = df4.apply(period_to_datetime, axis=1, time_mark='PDATE')
df3.columns = ['PDATE', 'METVALUE']
df5 = pd.merge(df4, df3, on='PDATE')
df5.pop('PERIODID')
df = pd.concat([df1,df2, df5], axis=0)
df['PDATE'] = pd.to_datetime(df['PDATE'])
df = df.rename(columns={'PDATE':g_sIndexCol, 'POWER': f'dpv_{g_sPowerFcCol}', 'METVALUE': f'dpv_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_dpv = df.drop_duplicates(g_sIndexCol)


DF = pd.merge(df_haifeng, df_lufeng, how='outer', on=g_sIndexCol)
DF = pd.merge(DF, df_pv, how='outer', on=g_sIndexCol)
DF = pd.merge(DF, df_dpv, how='outer', on=g_sIndexCol)

condition = (DF['Date'].dt.hour < 6) | (DF['Date'].dt.hour >= 18)
# 将满足条件的pv值置为0
pv_columns = list(filter(lambda x: 'pv' in x.lower(), DF.columns))
DF.loc[condition, pv_columns] = 0

DF['newEnergy_Pred'] = DF['haifeng_Pred'] + DF['lufeng_Pred'] + DF['pv_Pred'] + DF['dpv_Pred']
DF['newEnergy_True'] = DF['haifeng_True'] + DF['lufeng_True'] + DF['pv_True'] + DF['dpv_True']

cols = list(DF.columns)[1:]
DF[cols] = DF[cols].interpolate(limit=8, axis=0)
DF.to_csv('data/processed/gd_newEnergy_data_0728.csv')
a=1
