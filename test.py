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

# 数据拼接,POWER是预测值，后面那个是实际值
####中调风电(海风)
df1 = pd.read_excel('data/20250520中调风电.xls')
df2 = pd.read_excel('data/装机及新能源/中调风电2506.xls')
df = pd.concat([df1, df2], axis=0)
df['PDATE'] = pd.to_datetime(df['PDATE'])
df = df.rename(columns={'PDATE':g_sIndexCol, 'POWER': f'haifeng_{g_sPowerFcCol}', 'METVALUE': f'haifeng_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_haifeng = df.drop_duplicates(g_sIndexCol)


### 地调风电（陆风）
df1 = pd.read_excel('data/装机及新能源/地调风电2506.xls')
df2 = pd.read_excel('data/20250520地调风电.xls')
df = pd.concat([df1, df2], axis=0)
df['PDATE'] = pd.to_datetime(df['PDATE'])
df = df.rename(columns={'PDATE':g_sIndexCol, 'POWER': f'lufeng_{g_sPowerFcCol}', 'METVALUE': f'lufeng_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_lufeng = df.drop_duplicates(g_sIndexCol)


### 集中式光伏
df = pd.read_excel('data/装机及新能源/中调集中式光伏2506.xlsx')
df['日期'] = pd.to_datetime(df['日期'])
df = df.rename(columns={'日期':g_sIndexCol, '计划出力': f'pv_{g_sPowerFcCol}', '实际出力': f'pv_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_pv = df.drop_duplicates(g_sIndexCol)


### 分布式光伏
df1 = pd.read_excel('data/装机及新能源/分布式光伏2506.xls')
df2 = pd.read_excel('data/20250520分布式光伏.xls')
df = pd.concat([df1, df2], axis=0)
df['PDATE'] = pd.to_datetime(df['PDATE'])
df = df.rename(columns={'PDATE':g_sIndexCol, 'POWER': f'dpv_{g_sPowerFcCol}', 'METVALUE': f'dpv_{g_sPowerCol}'})
df = df.sort_values(g_sIndexCol)
df = df.reset_index(drop=True)
df_dpv = df.drop_duplicates(g_sIndexCol)


DF = pd.merge(df_haifeng, df_lufeng, how='outer', on=g_sIndexCol)
DF = pd.merge(DF, df_pv, how='outer', on=g_sIndexCol)
DF = pd.merge(DF, df_dpv, how='outer', on=g_sIndexCol)
DF['newEnergy_Pred'] = DF['haifeng_Pred'] + DF['lufeng_Pred'] + DF['pv_Pred'] + DF['dpv_Pred']
DF['newEnergy_True'] = DF['haifeng_True'] + DF['lufeng_True'] + DF['pv_True'] + DF['dpv_True']

DF['newEnergy1_Pred'] = DF['haifeng_Pred'] + DF['lufeng_Pred']  + DF['dpv_Pred']
DF['newEnergy1_True'] = DF['haifeng_True'] + DF['lufeng_True'] + DF['dpv_True']

DF.to_csv('data/processed/gd_newEnergy_data.csv')
a=1
