import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.util import Surv
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

# 读取数据
data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/fram2.csv"
data = pd.read_csv(data_path)
data['DEATH'] = data['DEATH'].astype(bool)

# 创建生存数据格式
surv_data = Surv.from_dataframe('DEATH', 'TIMEDTH', data)

# 特征和标签
X = data[['AGE', 'SEX', 'SYSBP', 'DIABP', 'TOTCHOL', 'BMI', 'GLUCOSE', 'CURSMOKE', 'BPMEDS']]
y = surv_data

# 分割数据集
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

# 训练Cox比例风险模型
cox_model = CoxPHSurvivalAnalysis()
cox_model.fit(X_train, y_train)

# 预测风险评分
cox_risk_scores_train = cox_model.predict(X_train)
cox_risk_scores_test = cox_model.predict(X_test)

# 对风险评分进行归一化处理
cox_scaler = MinMaxScaler()
cox_risk_scores_train_normalized = cox_scaler.fit_transform(cox_risk_scores_train.reshape(-1, 1)).flatten()
cox_risk_scores_test_normalized = cox_scaler.transform(cox_risk_scores_test.reshape(-1, 1)).flatten()

# 计算时间依赖C指数
cox_c_index = concordance_index_ipcw(y_train, y_test, cox_risk_scores_test_normalized)
print(f"Cox Time-dependent C-index: {cox_c_index}")

# 计算动态AUC
test_times = [y[1] for y in y_test]
min_time = np.min(test_times)
max_time = np.max(test_times)
times = np.linspace(min_time + 1, max_time - 1, 10)

print(f"Min time: {min_time}, Max time: {max_time}")
print(f"Times: {times}")

cox_auc_values, cox_mean_auc = cumulative_dynamic_auc(y_train, y_test, cox_risk_scores_test_normalized, times)
print(f"Cox Dynamic AUC at different times: {cox_auc_values}")
print(f"Cox Mean AUC: {cox_mean_auc}")

plt.figure()
plt.plot(times, cox_auc_values, color='navy', lw=2, label=f'Mean AUC = {cox_mean_auc:.2f}')
plt.xlabel('Time')
plt.ylabel('AUC')
plt.title('Cox Dynamic AUC over time')
plt.legend(loc='lower right')
plt.show()

# 计算Brier得分
cox_brier_score = brier_score_loss([event for event, _ in y_test], cox_risk_scores_test_normalized)
print(f"Cox Brier Score: {cox_brier_score}")


# 实现动态辨别指数（DDI）
def dynamic_discrimination_index(auc_values, times):
    time_diffs = np.diff(times)
    weighted_auc = np.sum((auc_values[:-1] + auc_values[1:]) / 2 * time_diffs)
    total_time = times[-1] - times[0]
    ddi = weighted_auc / total_time
    return ddi


cox_ddi = dynamic_discrimination_index(cox_auc_values, times)
print(f"Cox Dynamic Discrimination Index (DDI): {cox_ddi}")

# 计算Harrell's C-index
cox_harrell_c_index = concordance_index([y[1] for y in y_test], -cox_risk_scores_test, [y[0] for y in y_test])
print(f"Cox Harrell's C-index: {cox_harrell_c_index}")

# 计算常规C-index
event_times = np.array([y[1] for y in y_test])
event_observed = np.array([y[0] for y in y_test])
cox_c_index_conventional = concordance_index(event_times, -cox_risk_scores_test, event_observed)
print(f"Cox Conventional C-index: {cox_c_index_conventional}")

# 计算常规AUC和ROC曲线
fpr, tpr, thresholds = roc_curve([event for event, _ in y_test], cox_risk_scores_test_normalized)
cox_roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {cox_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cox ROC Curve')
plt.legend(loc='lower right')
plt.show()
print(f"Cox ROC AUC: {cox_roc_auc}")

# 使用逻辑回归作为基线模型
log_reg = LogisticRegression()
y_train_binary = (y_train['DEATH'] == True).astype(int)
y_test_binary = (y_test['DEATH'] == True).astype(int)
log_reg.fit(X_train, y_train_binary)

# 预测基线模型的风险评分
baseline_risk_scores_test = log_reg.predict_proba(X_test)[:, 1]


# 计算NRI和IDI
def calculate_nri_idi(y_true, baseline_scores, new_scores):
    # 计算新的风险评分的事件和非事件
    events = y_true == 1
    nonevents = y_true == 0
    events_baseline_scores = baseline_scores[events]
    events_new_scores = new_scores[events]
    nonevents_baseline_scores = baseline_scores[nonevents]
    nonevents_new_scores = new_scores[nonevents]

    # 计算事件的重分类情况
    event_reclassified_up = np.sum(events_new_scores > events_baseline_scores) / len(events_baseline_scores)
    event_reclassified_down = np.sum(events_new_scores < events_baseline_scores) / len(events_baseline_scores)

    # 计算非事件的重分类情况
    nonevent_reclassified_up = np.sum(nonevents_new_scores > nonevents_baseline_scores) / len(nonevents_baseline_scores)
    nonevent_reclassified_down = np.sum(nonevents_new_scores < nonevents_baseline_scores) / len(
        nonevents_baseline_scores)

    # 计算NRI
    nri = (event_reclassified_up - event_reclassified_down) - (nonevent_reclassified_up - nonevent_reclassified_down)

    # 计算IDI
    idi = (np.mean(events_new_scores) - np.mean(events_baseline_scores)) - (
                np.mean(nonevents_new_scores) - np.mean(nonevents_baseline_scores))

    return nri, idi


cox_nri, cox_idi = calculate_nri_idi(y_test_binary, baseline_risk_scores_test, cox_risk_scores_test_normalized)
print(f"Cox Net Reclassification Improvement (NRI): {cox_nri}")
print(f"Cox Integrated Discrimination Improvement (IDI): {cox_idi}")
