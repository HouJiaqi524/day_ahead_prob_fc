---
header-includes: |
  \usepackage{ctex}
---
[toc]

# 文档修改记录

| 日期     | 内容                                       | 修改人 |
| -------- | ------------------------------------------ | ------ |
| 20250811 | 日前新能源区间预测和专项备用计算，算法初版 | 侯嘉琪 |

# 算法调用

* 通过 /p: 和 /e: 分别设置输入、输出目录，目录下存放一下列出的输入数据、输出数据的.txt

a.--用法：Command /p:InputPath  /e:OutputPath

b.--例如：./scene /p:/home/tsintergy/hjq/long_term_prob_fc/FILE_IN /e:/home/tsintergy/hjq/long_term_prob_fc/FILE_OU

* 输出的debug文件名为“Debug.e”；“进度信息.e”会显示执行进度信息，将存放在输出目录路径下。

# 输入数据（Input）

> **缺数用 null/Null/NULL 填充**

## 系统声明 (每个txt均包含)

```
<!Type=日前新能源区间预测和专项备用计算 Grid=广东 Time='2025-07-26 14:31:35' !>
```

| 属性 | 说明           | 类型   | 取值                | 默认值  | 单位 |
| ---- | -------------- | ------ | ------------------- | ------- | ---- |
| Grid | 电网名称       | string |                     | Default |      |
| Type | 数据文件类型   | string |                     |         |      |
| Time | 数据文件时间戳 | string | yyyy-MM-dd HH:mm:ss |         |      |

## 控制参数表 ControlParameter.txt

```xml
<ControlParameter>
@  ParamID  ParamName  Value
//  参数ID  参数名称  参数值
#  SceneNumber  场景数  1000
#  Confidence 置信度 [80,90,95]
…
</ControlParameter>
```

| 字段      | 说明     | 类型   | 取值 | 默认值 | 单位 |
| --------- | -------- | ------ | ---- | ------ | ---- |
| ParamID   | 参数 ID  | string |      |        |      |
| ParamName | 参数名称 | string |      |        |      |
| Value     | 取值     | string |      |        |      |

> 注意：下表给出具体数据项。

| 属性 ID         | 属性名称               | 取值                                                                                  | 默认值     | 单位   | 备注                                                |
| --------------- | ---------------------- | ------------------------------------------------------------------------------------- | ---------- | ------ | --------------------------------------------------- |
| Confidence      | 置信度                 | 不含空格的列表。（如需要计算80%，90%，95%置信度的区间估计，则Confidence为[80,90,95]） | [80,90,95] |        | 前端用户配置                                        |
| SceneNumber     | 蒙特卡洛采样个数       | int, (500,10000)                                                                      | 1000       |        |                                                     |
| ForecastType    | 预测维度               | 可选：日、周                                                                          |            |        |                                                     |
| ForecastSDay    | 预测开始日             | YYYY-MM-DD                                                                            |            |        | 前端用户配置                                        |
| ForecastEDay    | 预测结束日             | YYYY-MM-DD                                                                            |            |        | 前端用户配置                                        |
| PointNumber     | 历史数据的日时刻点数   | 如24、96                                                                              | 96         |        |                                                     |
| DedicPosResCost | 专项正备用单位成本     | 正数                                                                                  | 1          | 元/MWh | 前端用户配置                                        |
| DedicNegResCost | 专项负备用单位成本     | 正数                                                                                  | 1          | 元/MWh | 前端用户配置                                        |
| SecurPosResCost | 安全正备用裕度单位成本 | 正数                                                                                  | 15         | 元/MWh | 前端用户配置                                        |
| SecurNegResCost | 安全负备用裕度单位成本 | 正数                                                                                  | 5          | 元/MWh | 前端用户配置                                        |
| IsPostEvaluate  | 是否启动后评估配置     | bool                                                                                  | 0,1        |        | 前端用户配置，当配置1，需输入预测日段的实际功率信息 |
| SampleSelect    | 样本选择模式           | 1；2；3                                                                               | 1          |        | 前端用户配置，如选择3，需再配置选择的样本           |

> **样本选择模式：**   1： 近3个月样本；2：算法根据预报功率选择前60个相似样本 ；3：用户通过前端挑选样本并写入DateInfo/TrainDate（注，建议挑选30日以上的样本，至少10天）

## 基础信息（目录：BasicInfo）

### 装机容量 Capacity.txt

| 字段         | 说明                    | 类型   | 取值        | 默认值 | 单位 |
| ------------ | ----------------------- | ------ | ----------- | ------ | ---- |
| Date         | 日期                    | string | YYYY-MM--DD |        |      |
| OnshoreWind  | 陆上风电                | double |             |        | MW   |
| OffshoreWind | 海上风电                | double |             |        | MW   |
| WindPower    | 风电                    | double |             |        | MW   |
| CPV          | 集中式光伏发电          | double |             |        | MW   |
| DPV          | 分布式光伏发电          | double |             |        | MW   |
| PV           | 光伏发电                | double |             |        | MW   |
| NewEnergy    | 新能源（风电+光伏发电） | double |             |        | MW   |

> **备注：** 日期至少是月颗粒度，按截至月底的装机容量，日颗粒度更好。需包含待预测日期所在月的装机容量。

### 气象位点基本信息 StationInfo.txt

| 字段     | 说明                        | 类型   | 取值 | 默认值 | 单位 |
| -------- | --------------------------- | ------ | ---- | ------ | ---- |
| Station  | 气象位点（气象站点/网格点） | string |      |        |      |
| Lon      | 经度                        | double |      |        |      |
| Lat      | 纬度                        | double |      |        |      |
| Province | 所属省份                    | string |      |        |      |
| City     | 所属地市                    | string |      |        |      |
| District | 所属区县                    | string |      |        |      |

### 对象和气象映射表 MapTable

| 字段        | 说明             | 类型   | 取值                                                               | 默认值 | 单位 |
| ----------- | ---------------- | ------ | ------------------------------------------------------------------ | ------ | ---- |
| Object      | 预测对象         | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy      |        |      |
| ObjectName  | 预测对象         | string | 陆上风电，海上风电，集中式光伏，分布式光伏，风电，光伏发电，新能源 |        |      |
| StationList | 关联气象位点列表 | string | 不含空格的列表，如[57736,64279]                                    |        |      |

> **备注：** 具体映射关系可有算法测算匹配，离线提供。映射关系可阶段性保持不变。

### 三峰两谷时刻 TurningPointInfo

| 字段          | 说明     | 类型   | 取值       | 默认值 | 单位 |
| ------------- | -------- | ------ | ---------- | ------ | ---- |
| Date          | 日期     | string | YYYY-MM-DD |        |      |
| DawnValley    | 凌晨低谷 | string | hh:mm      | 5:00   |      |
| MorningPeak   | 早晨高峰 | string | hh:mm      | 11:00  |      |
| NoonValley    | 午间低谷 | string | hh:mm      | 12:00  |      |
| AfternoonPeak | 下午高峰 | string | hh:mm      | 17:00  |      |
| EveningPeak   | 晚间高峰 | string | hh:mm      | 19:00  |      |

## 日期信息 （目录： DateInfo）

### 节假日 HolidayInfo

| 字段         | 说明           | 类型   | 取值          | 默认值 | 单位 |
| ------------ | -------------- | ------ | ------------- | ------ | ---- |
| Date         | 日期           | string | YYYY-MM-DD    |        |      |
| HolidayType  | 节假日类型     | int    | 1,2,3,4,5,6,7 | 5:00   |      |
| DaysAhead    | 该日前休假天数 | int    |               |        |      |
| DaysAfter    | 该日后休假天数 | int    |               |        |      |
| IsHolidayDay | 是否为节日当天 | bool   | 0,1           |        |      |

> **备注：** 1（元旦当天）;2（春节当天）;3（清明节当天）;4（劳动节当天）;5（端午节当天）;6（中秋节当天）;7（国庆节当天）。

### 调休补班日 AdjustedWorkday

| 字段 | 说明 | 类型   | 取值       | 默认值 | 单位 |
| ---- | ---- | ------ | ---------- | ------ | ---- |
| Date | 日期 | string | YYYY-MM-DD |        |      |

### 特殊异常日 DateNotIncluded

| 字段            | 说明             | 类型                                                                        | 取值                                           | 默认值 | 单位 |
| --------------- | ---------------- | --------------------------------------------------------------------------- | ---------------------------------------------- | ------ | ---- |
| Date            | 日期             | string                                                                      | YYYY-MM-DD                                     |        |      |
| Cause           | 数据异常原因     | string                                                                      | 简短的真实原因。如台风影响，2000MW海风风机停运 |        |      |
| AffectedObjects | 受影响的预测对象 | 不含空格的列表，如[OnshoreWind,OffshoreWind,CPV,DPV,WindPower,PV,NewEnergy] |                                                |        |      |

### 训练样本 TrainDate

| 字段   | 说明     | 类型   | 取值                                                          | 默认值 | 单位 |
| ------ | -------- | ------ | ------------------------------------------------------------- | ------ | ---- |
| Date   | 日期     | string | YYYY-MM-DD                                                    |        |      |
| Object | 预测对象 | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy |        |      |

> **备注：** 当ControlParameter中SampleSelect=3时，该表不能为空。

## 运行数据（目录：OperationData）

> **备注：** 实际功率、预测功率、数值天气预报需写入近2年的数据；需包含ForecastSDay~ForecastEDay的预测功率、数值天气预报。

### 实际功率 HisPower

| 字段   | 说明              | 类型   | 取值                                                                        | 默认值 | 单位 |
| ------ | ----------------- | ------ | --------------------------------------------------------------------------- | ------ | ---- |
| Date   | 日期              | string | YYYY-MM-DD                                                                  |        |      |
| Object | 预测对象          | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy，NewEnergyAdd |        |      |
| T0000  | 时刻点0:00出力值  | double |                                                                             |        | MW   |
| T0015  | 时刻点0:15出力值  | double |                                                                             |        | MW   |
| T0030  | 时刻点0:30出力值  | double |                                                                             |        | MW   |
| ……   | ……              | ……   |                                                                             |        | …… |
| T2330  | 时刻点23:30出力值 | double |                                                                             |        | MW   |
| T2345  | 时刻点23:45出力值 | double |                                                                             |        | MW   |

### 预报功率 FcPower

| 字段   | 说明              | 类型   | 取值                                                                        | 默认值 | 单位 |
| ------ | ----------------- | ------ | --------------------------------------------------------------------------- | ------ | ---- |
| Date   | 日期              | string | YYYY-MM-DD                                                                  |        |      |
| Object | 预测对象          | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy，NewEnergyAdd |        |      |
| T0000  | 时刻点0:00出力值  | double |                                                                             |        | MW   |
| T0015  | 时刻点0:15出力值  | double |                                                                             |        | MW   |
| T0030  | 时刻点0:30出力值  | double |                                                                             |        | MW   |
| ……   | ……              | ……   |                                                                             |        | …… |
| T2330  | 时刻点23:30出力值 | double |                                                                             |        | MW   |
| T2345  | 时刻点23:45出力值 | double |                                                                             |        | MW   |

## 数值天气预报 （目录：NWP）

### Hres预报_分起报时间分文件存储 Hres_，例如Hres_2020101012.txt

| 字段                | 说明                         | 类型   | 取值             | 默认值 | 单位 |
| ------------------- | ---------------------------- | ------ | ---------------- | ------ | ---- |
| Date                | 日期时刻                     | string | YYYY-MM-DD_hh:mm |        |      |
| {Feature}_{Station} | {气象指标}_{气象站点/网格点} | string |                  |        |      |

> **气象类型:**   Feature取如下指标，100mWS:100m风速；ssr: 地表净短波辐射。Station为广东域内所有站点，参考【气象位点基本信息 StationInfo.txt】。

样例：

```
<NWPHis>
@	Date	t2m_59090	ssr_acc_59090	u100m_59090  v100m_59090 ……
#	2024-06-01_00:00:00	297.57	0.0	-3.81	-3.49	298.89	0.0	-1.3……
#	2024-06-01_00:15:00	297.54048009227114	0.0	-3.745522837456452	-3.6129174890347726 …
……
</NWPHis>
```

# 输出数据 (Output)

### 区间预测结果 IntervalInfo

| 字段     | 说明               | 类型   | 取值                                                                        | 备注                                   | 单位 |
| -------- | ------------------ | ------ | --------------------------------------------------------------------------- | -------------------------------------- | ---- |
| DateTime | 日期时间           | string | YYYY-MM-DD_hh:mm                                                            |                                        |      |
| Object   | 预测对象           | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy，NewEnergyAdd |                                        |      |
| Hispower | 实际功率           | string |                                                                             | 后评估配置下输出，从输入的实际功率搬出 | MW   |
| Fcpower  | 预测功率           | string |                                                                             | 从输入的预测功率搬出                   | MW   |
| Low80    | 80%置信下界        | string |                                                                             |                                        | MW   |
| Up80     | 80%置信上界        | string |                                                                             |                                        | MW   |
| ……     | ……               | ……   |                                                                             |                                        | …… |
| Low95    | 95%置信下界        | string |                                                                             |                                        | MW   |
| Up95     | 95%置信上界        | string |                                                                             |                                        | MW   |
| Mean     | 调整的预测功率均值 | string |                                                                             | 概率模型修正的预测值                   | MW   |
| Standard | 预测功率的标准差   | string |                                                                             |                                        | MW   |

### 新能源专项备用相关 DedicatedRERes

| 字段            | 说明                           | 类型   | 取值                                                                        | 备注 | 单位 |
| --------------- | ------------------------------ | ------ | --------------------------------------------------------------------------- | ---- | ---- |
| DateTime        | 日期时间                       | string | YYYY-MM-DD_hh:mm                                                            |      |      |
| Object          | 预测对象                       | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy，NewEnergyAdd |      |      |
| PosRERes        | 新能源专项正备用               | string |                                                                             |      | MW   |
| NegRERes        | 新能源专项负备用               | string |                                                                             |      | MW   |
| PosREResConf    | 新能源专项正备用对应的置信度   | string |                                                                             |      |      |
| NegREResConf    | 新能源专项负备用对应的置信度   | string |                                                                             |      |      |
| PosREResCost    | 新能源专项正备用成本           | string |                                                                             |      | 元/h |
| NegREResCost    | 新能源专项负备用成本           | string |                                                                             |      | 元/h |
| PosSecurRes     | 占用安全正备用裕度的期望容量   | string |                                                                             |      | 元/h |
| NegSecurRes     | 占用安全负备用裕度的期望容量   | string |                                                                             |      | MW   |
| PosSecurResCost | 占用安全正备用裕度的期望成本   | string |                                                                             |      | 元/h |
| NegSecurResCost | 占用安全负备用裕度的期望成本   | string |                                                                             |      | 元/h |
| PosResCost      | 抵御新能源波动正备用的期望成本 | string |                                                                             |      | 元/h |
| NegResCost      | 抵御新能源波动负备用的期望成本 | string |                                                                             |      | 元/h |

### 后评估增补 AdditionalPostEvaluate

| 字段                | 说明                           | 类型   | 取值                                                                        | 备注 | 单位 |
| ------------------- | ------------------------------ | ------ | --------------------------------------------------------------------------- | ---- | ---- |
| DateTime            | 日期时间                       | string | YYYY-MM-DD_hh:mm                                                            |      |      |
| Object              | 预测对象                       | string | OnshoreWind，OffshoreWind，CPV，DPV，WindPower，PV，NewEnergy，NewEnergyAdd |      |      |
| RealPosSecurRes     | 占用安全正备用裕度的实际容量   | double |                                                                             |      | MW   |
| RealNegSecurRes     | 占用安全负备用裕度的实际容量   | double |                                                                             |      | MW   |
| RealPosSecurResCost | 占用安全正备用裕度的实际成本   | double |                                                                             |      | 元/h |
| RealNegSecurResCost | 占用安全负备用裕度的实际成本   | double |                                                                             |      | 元/h |
| RealPosResCost      | 抵御新能源波动正备用实际总成本 | double |                                                                             |      | 元/h |
| RealNegResCost      | 抵御新能源波动负备用实际总成本 | double |                                                                             |      | 元/h |
| IsMoreThan4000      | 占用安全裕度是否超过4000       | bool   | 0，1                                                                        |      |      |
