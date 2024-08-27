import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from fancyimpute import IterativeImputer

# 加载数据集
data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/frmgham2.csv"
data_path2 = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/fram2.csv"
data = pd.read_csv(data_path)
data2 = pd.read_csv(data_path2)
# 查看数据集中缺失值情况
missing_values = data.isnull().sum()
print("缺失值情况：\n", missing_values)

# 填补缺失值，可以根据需要选择合适的填补方法
# 多重插补
# 使用 IterativeImputer 进行多重插补，相当于MICE
imputer = IterativeImputer(max_iter=10, random_state=0)
data_imputed = imputer.fit_transform(data)

# 将插补后的数据转换回DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# 查看处理后的数据集
print("处理后的数据集：\n", data_imputed.head())
missing_values1 = data_imputed.isnull().sum()
print("缺失值情况：\n", missing_values1)

# 检查事件时间和事件状态的分布
event_times = ['TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD', 'TIMEHYP', 'TIMEDTH']
event_status = ['ANGINA', 'HOSPMI', 'MI_FCHD', 'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN', 'DEATH']

# 打印事件时间和状态的描述统计
print(data_imputed[event_times + event_status].describe())

# 判断右删失数据
# 使用 TIMEMI 作为时间变量，HOSPMI 作为事件变量
time_col = 'TIMEMI'
event_col = 'HOSPMI'

# 筛选出删失数据
censored_data = data_imputed[data_imputed[event_col] == 0]
print("\n右删失数据的数量：", len(censored_data))
print("右删失数据示例：")
print(censored_data[[time_col, event_col]].head())

# 统计右删失数据的比例
total_rows = len(data_imputed)
censored_count = len(censored_data)
censored_ratio = censored_count / total_rows

print(f"\n右删失数据比例：{censored_ratio:.2%}")

# 使用 TIMEMI 作为时间变量，HOSPMI 作为事件变量
time_col = 'TIMEMI'
event_col = 'HOSPMI'

# 筛选出右删失数据
censored_data = data_imputed[data_imputed[event_col] == 0]

# 查看右删失数据的数量和分布
print("\n右删失数据的数量：", len(censored_data))
print("右删失数据示例：")
print(censored_data[[time_col, event_col]].head())

# 绘制右删失数据的时间分布
plt.figure(figsize=(10, 6))
sns.histplot(censored_data[time_col], bins=30, kde=True)
plt.title('Distribution of Censored Data (TIMEMI)')
plt.xlabel('Time (days)')
plt.ylabel('Frequency')
plt.show()

# 绘制右删失数据的密度分布
plt.figure(figsize=(10, 6))
sns.kdeplot(censored_data[time_col], shade=True)
plt.title('Density Distribution of Censored Data (TIMEMI)')
plt.xlabel('Time (days)')
plt.ylabel('Density')
plt.show()

# 创建KaplanMeierFitter对象
kmf = KaplanMeierFitter()
data_imputed['time'] = data_imputed['TIMEMI']  # 从基线到第一次住院心肌梗死事件的时间
data_imputed['status'] = data_imputed['HOSPMI']  # 事件状态，1表示事件发生，0表示删失
# 拟合数据
kmf.fit(durations=data_imputed['time'], event_observed=data_imputed['status'])

# 打印生存概率
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve for TIMEMI')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.show()

# 计算Cox模型以估计删失时间的生存概率
cph = CoxPHFitter()
data_imputed['time'] = data_imputed['TIMEMI']  # 从基线到第一次住院心肌梗死事件的时间
data_imputed['status'] = data_imputed['HOSPMI']  # 事件状态，1表示事件发生，0表示删失
cph.fit(data_imputed[['time', 'status', 'TOTCHOL', 'AGE', 'SEX', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE']], 'time', event_col='status')

# 计算生存概率
data_imputed['surv_prob'] = cph.predict_survival_function(data_imputed[['TOTCHOL', 'AGE', 'SEX', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE']], times=data_imputed['time']).T.iloc[:, 0]

# 定义IPCW加权函数
def ipcw_weighted_roc(data, time_points):
    aucs = []
    for t in time_points:
        subset_data = data[data['time'] >= t]
        if len(subset_data) == 0:
            continue

        y_true = subset_data['status']
        y_scores = subset_data['TOTCHOL']  # 使用TOTCHOL作为标记变量
        weights = 1.0 / subset_data['surv_prob']

        # 计算加权的AUC
        auc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        aucs.append(auc)
    return aucs

# 计算IPCW加权AUC
time_points = [1000, 2000, 3000]
ipcw_aucs = ipcw_weighted_roc(data_imputed, time_points)
print("IPCW AUCs:", ipcw_aucs)

# 最近邻估计量函数
def nearest_neighbor_auc(data, time_points, k):
    aucs = []
    for t in time_points:
        events = data[(data['time'] <= t) & (data['status'] == 1)]
        censored = data[data['time'] > t]

        if len(events) == 0 or len(censored) == 0:
            continue

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(censored[['TOTCHOL']].values)
        auc = 0

        for _, event in events.iterrows():
            distances, indices = nn.kneighbors([[event['TOTCHOL']]])
            nearest_censored = censored.iloc[indices[0]]
            auc += (event['TOTCHOL'] > nearest_censored['TOTCHOL']).mean()

        auc /= len(events)
        aucs.append(auc)
    return aucs

# 计算最近邻估计量AUC
#time_points = [1000, 2000, 3000,4000,5000,6000]
#nne_aucs = nearest_neighbor_auc(data_imputed, time_points, k=5000)
#print("NNE AUCs:", nne_aucs)

# 定义CIPCW加权函数
def cipcw_weighted_roc(data, time_points):
    aucs = []
    for t in time_points:
        subset_data = data[data['time'] >= t]
        if len(subset_data) == 0:
            continue

        y_true = subset_data['status']
        y_scores = subset_data['TOTCHOL']  # 使用TOTCHOL作为标记变量
        weights = 1.0 / cph.predict_survival_function(subset_data[['TOTCHOL', 'AGE', 'SEX', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE']], times=[t]).T.iloc[:, 0]

        # 计算加权的AUC
        auc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        aucs.append(auc)
    return aucs

# 计算CIPCW加权AUC
cipcw_aucs = cipcw_weighted_roc(data_imputed, time_points)
print("CIPCW AUCs:", cipcw_aucs)
