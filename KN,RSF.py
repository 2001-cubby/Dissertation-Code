import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lifelines import KaplanMeierFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import numpy as np
from sklearn.metrics import roc_auc_score

# 加载数据集
data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/frmgham2.csv"
data = pd.read_csv(data_path)

# 填补缺失值
imputer = IterativeImputer(max_iter=10, random_state=0)
data_imputed = imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# 使用 TIMEMI 作为时间变量，HOSPMI 作为事件变量
time_col = 'TIMEMI'
event_col = 'HOSPMI'

# 转换为适合生存分析的格式
data_imputed['time'] = data_imputed[time_col]
data_imputed['status'] = data_imputed[event_col].astype(bool)

# 计算Kaplan-Meier生存概率
kmf = KaplanMeierFitter()
kmf.fit(durations=data_imputed['time'], event_observed=~data_imputed['status'])
data_imputed['surv_prob_km'] = kmf.survival_function_at_times(data_imputed['time']).values

# 设置生存概率的最小值阈值以避免无穷大的权重
epsilon = 1e-5
data_imputed['surv_prob_km'] = np.maximum(data_imputed['surv_prob_km'], epsilon)

# 准备RSF模型生存概率
X = data_imputed[['TOTCHOL', 'AGE', 'SEX', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE']]
y = Surv.from_dataframe('status', 'time', data_imputed)

rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=0)
rsf.fit(X, y)

# 计算基于RSF的生存概率
surv_prob_rsf = rsf.predict_survival_function(X)
surv_prob_rsf_values = np.array([fn(data_imputed['time'].values) for fn in surv_prob_rsf]).T
surv_prob_rsf_values = np.maximum(surv_prob_rsf_values, epsilon)
data_imputed['surv_prob_rsf'] = surv_prob_rsf_values[np.arange(len(data_imputed)), np.searchsorted(data_imputed['time'].values, data_imputed['time'].values)]
# 定义IPCW加权函数
def ipcw_weighted_roc(data, time_points, survival_probs):
    aucs = []
    for t in time_points:
        subset_data = data[data['time'] >= t]
        if len(subset_data) == 0:
            continue
        y_true = subset_data['status']
        y_scores = subset_data['TOTCHOL']  # 使用TOTCHOL作为标记变量
        weights = 1.0 / subset_data[survival_probs]
        auc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        aucs.append(auc)
    return aucs

# 定义CIPCW加权函数
def cipcw_weighted_roc(data, time_points, survival_probs):
    aucs = []
    for t in time_points:
        subset_data = data[data['time'] >= t]
        if len(subset_data) == 0:
            continue
        y_true = subset_data['status']
        y_scores = subset_data['TOTCHOL']  # 使用TOTCHOL作为标记变量
        weights = 1.0 / subset_data[survival_probs]
        auc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        aucs.append(auc)
    return aucs
# 设置时间点
time_points = [1000, 2000, 3000]

# 计算IPCW加权AUC
ipcw_aucs_km = ipcw_weighted_roc(data_imputed, time_points, 'surv_prob_km')
print("IPCW(KM) AUCs:", ipcw_aucs_km)

ipcw_aucs_rsf = ipcw_weighted_roc(data_imputed, time_points, 'surv_prob_rsf')
print("IPCW(RSF) AUCs:", ipcw_aucs_rsf)

# 计算CIPCW加权AUC
cipcw_aucs_km = cipcw_weighted_roc(data_imputed, time_points, 'surv_prob_km')
print("CIPCW(KM) AUCs:", cipcw_aucs_km)

cipcw_aucs_rsf = cipcw_weighted_roc(data_imputed, time_points, 'surv_prob_rsf')
print("CIPCW(RSF) AUCs:", cipcw_aucs_rsf)
