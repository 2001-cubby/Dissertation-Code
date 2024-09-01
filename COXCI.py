import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.util import Surv
from sklearn.preprocessing import MinMaxScaler
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 适配Cox比例风险模型
cox_model = CoxPHSurvivalAnalysis()
cox_model.fit(X_train, y_train)

# 预测风险评分
risk_scores_train = cox_model.predict(X_train)
risk_scores_test = cox_model.predict(X_test)

# 对风险评分进行归一化处理
scaler = MinMaxScaler()
risk_scores_train_normalized = scaler.fit_transform(risk_scores_train.reshape(-1, 1)).flatten()
risk_scores_test_normalized = scaler.transform(risk_scores_test.reshape(-1, 1)).flatten()

# 计算标准误差和置信区间
def calculate_standard_error(value, n):
    se = np.sqrt((value * (1 - value)) / n)
    return se

def calculate_ci(value, se, alpha=0.05):
    z = 1.96  # 对应95%置信区间
    lower = max(0, value - z * se)
    upper = min(1, value + z * se)
    return lower, upper

# 计算时间依赖C指数
c_index_ipcw = concordance_index_ipcw(y_train, y_test, risk_scores_test_normalized)
c_index_value = c_index_ipcw[0]
c_index_se = calculate_standard_error(c_index_value, len(y_test))
c_index_ci = calculate_ci(c_index_value, c_index_se)
print(f"Time-dependent C-index: {c_index_value:.4f} (95% CI: {c_index_ci[0]:.4f} - {c_index_ci[1]:.4f})")

# 计算动态AUC
test_times = [y[1] for y in y_test]
min_time = np.min(test_times)
max_time = np.max(test_times)
times = np.linspace(min_time + 1, max_time - 1, 10)

auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_test_normalized, times)
mean_auc_se = calculate_standard_error(mean_auc, len(y_test))
mean_auc_ci = calculate_ci(mean_auc, mean_auc_se)
print(f"Dynamic AUC at different times: {auc_values}")
print(f"Mean Dynamic AUC: {mean_auc:.4f} (95% CI: {mean_auc_ci[0]:.4f} - {mean_auc_ci[1]:.4f})")

plt.figure()
plt.plot(times, auc_values, color='navy', lw=2, label=f'Mean AUC = {mean_auc:.2f}')
plt.xlabel('Time')
plt.ylabel('AUC')
plt.title('Dynamic AUC over time')
plt.legend(loc='lower right')
plt.show()

# 实现动态辨别指数（DDI）
def dynamic_discrimination_index(auc_values, times):
    time_diffs = np.diff(times)
    weighted_auc = np.sum((auc_values[:-1] + auc_values[1:]) / 2 * time_diffs)
    total_time = times[-1] - times[0]
    ddi = weighted_auc / total_time
    return ddi

ddi = dynamic_discrimination_index(auc_values, times)
ddi_se = calculate_standard_error(ddi, len(y_test))
ddi_ci = calculate_ci(ddi, ddi_se)
print(f"Dynamic Discrimination Index (DDI): {ddi:.4f} (95% CI: {ddi_ci[0]:.4f} - {ddi_ci[1]:.4f})")

# 计算Harrell's C-index
harrell_c_index = concordance_index([y[1] for y in y_test], -risk_scores_test, [y[0] for y in y_test])
harrell_c_index_se = calculate_standard_error(harrell_c_index, len(y_test))
harrell_c_index_ci = calculate_ci(harrell_c_index, harrell_c_index_se)
print(f"Harrell's C-index: {harrell_c_index:.4f} (95% CI: {harrell_c_index_ci[0]:.4f} - {harrell_c_index_ci[1]:.4f})")

# 计算常规C-index
event_times = np.array([y[1] for y in y_test])
event_observed = np.array([y[0] for y in y_test])
c_index_conventional = concordance_index(event_times, -risk_scores_test, event_observed)
c_index_conventional_se = calculate_standard_error(c_index_conventional, len(y_test))
c_index_conventional_ci = calculate_ci(c_index_conventional, c_index_conventional_se)
print(f"Conventional C-index: {c_index_conventional:.4f} (95% CI: {c_index_conventional_ci[0]:.4f} - {c_index_conventional_ci[1]:.4f})")

# 计算常规AUC和ROC曲线
fpr, tpr, thresholds = roc_curve([event for event, _ in y_test], risk_scores_test_normalized)
roc_auc = auc(fpr, tpr)
roc_auc_se = calculate_standard_error(roc_auc, len(y_test))
roc_auc_ci = calculate_ci(roc_auc, roc_auc_se)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('COX ROC Curve')
plt.legend(loc='lower right')
plt.show()
print(f"ROC AUC: {roc_auc:.4f} (95% CI: {roc_auc_ci[0]:.4f} - {roc_auc_ci[1]:.4f})")

