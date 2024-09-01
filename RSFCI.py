import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index


# 计算标准误差的函数
def calculate_standard_error(value, n):
    """
    计算标准误差
    """
    se = np.sqrt((value * (1 - value)) / n)
    return se


# 计算置信区间的函数
def calculate_ci(value, se, alpha=0.05):
    """
    计算置信区间
    """
    z = 1.96  # 对应于95%的置信区间
    lower = value - z * se
    upper = value + z * se
    # 确保CI在[0,1]范围内
    lower = max(0, lower)
    upper = min(1, upper)
    return lower, upper


# 计算NRI和IDI的置信区间
def nri_idi_ci(nri, idi, n, alpha=0.05):
    """
    计算NRI和IDI的置信区间
    """
    se_nri = np.sqrt((nri * (1 - nri)) / n)
    se_idi = np.sqrt((idi * (1 - idi)) / n)

    z = 1.96  # 对应于95%的置信区间

    lower_nri = nri - z * se_nri
    upper_nri = nri + z * se_nri
    lower_idi = idi - z * se_idi
    upper_idi = idi + z * se_idi

    # 确保CI在合理范围内
    lower_nri = max(-1, lower_nri)
    upper_nri = min(1, upper_nri)
    lower_idi = max(-1, lower_idi)
    upper_idi = min(1, upper_idi)

    return (lower_nri, upper_nri), (lower_idi, upper_idi)


# 读取数据
data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/fram2.csv"
data = pd.read_csv(data_path)
data['DEATH'] = data['DEATH'].astype(bool)

# 创建生存数据格式
y = Surv.from_dataframe('DEATH', 'TIMEDTH', data)

# 选择特征和标签
X = data[['AGE', 'SEX', 'SYSBP', 'DIABP', 'TOTCHOL', 'BMI', 'GLUCOSE', 'CURSMOKE', 'BPMEDS']]

# 分割数据集
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

# 训练随机生存森林模型
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X_train, y_train)

# 预测风险评分
rsf_risk_scores_train = rsf.predict(X_train)
rsf_risk_scores_test = rsf.predict(X_test)

# 对风险评分进行归一化处理
rsf_scaler = MinMaxScaler()
rsf_risk_scores_train_normalized = rsf_scaler.fit_transform(rsf_risk_scores_train.reshape(-1, 1)).flatten()
rsf_risk_scores_test_normalized = rsf_scaler.transform(rsf_risk_scores_test.reshape(-1, 1)).flatten()

# 确定样本量
n_test = len(y_test)


# 计算C-index的置信区间（常规C-index）
def compute_c_index_ci(y_test, risk_scores, n):
    """
    计算常规C-index及其置信区间
    """
    event_times = np.array([y[1] for y in y_test])
    event_observed = np.array([y[0] for y in y_test])
    c_index = concordance_index(event_times, -risk_scores, event_observed)
    se = calculate_standard_error(c_index, n)
    ci = calculate_ci(c_index, se)
    return c_index, ci


# 计算时间依赖C-index的置信区间
def compute_time_dependent_c_index_ci(y_train, y_test, risk_scores, n):
    """
    计算时间依赖C-index及其置信区间
    """
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, risk_scores)
    c_index = c_index_ipcw[0]  # IPCW C-index的点估计
    # 注意：计算时间依赖C-index的标准误差较为复杂，这里简化处理
    se = calculate_standard_error(c_index, n)
    ci = calculate_ci(c_index, se)
    return c_index, ci


# 计算Harrell's C-index的置信区间
def compute_harrell_c_index_ci(y_test, risk_scores, n):
    """
    计算Harrell's C-index及其置信区间
    """
    event_times = np.array([y[1] for y in y_test])
    event_observed = np.array([y[0] for y in y_test])
    harrell_c_index = concordance_index(event_times, -risk_scores, event_observed)
    se = calculate_standard_error(harrell_c_index, n)
    ci = calculate_ci(harrell_c_index, se)
    return harrell_c_index, ci


# 计算ROC AUC及其置信区间
def compute_roc_auc_ci(y_test, risk_scores, n):
    """
    计算ROC AUC及其置信区间
    """
    # 将y_test转换为二进制
    y_test_binary = np.array([event for event, _ in y_test]).astype(int)
    roc_auc = roc_auc_score(y_test_binary, risk_scores)
    se = calculate_standard_error(roc_auc, n)
    ci = calculate_ci(roc_auc, se)
    return roc_auc, ci


# 计算动态AUC及其置信区间
def compute_dynamic_auc_ci(y_train, y_test, risk_scores, times, n):
    """
    计算动态AUC及其置信区间
    """
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    # 计算每个时间点的CI
    se_auc = calculate_standard_error(auc_values, n)
    ci_auc = [calculate_ci(auc, se) for auc, se in zip(auc_values, se_auc)]
    # 计算mean AUC的CI
    se_mean_auc = calculate_standard_error(mean_auc, n)
    ci_mean_auc = calculate_ci(mean_auc, se_mean_auc)
    return auc_values, mean_auc, ci_auc, ci_mean_auc


# 计算动态辨别指数（DDI）的置信区间
def compute_ddi_ci(ddi, n):
    """
    计算DDI及其置信区间
    """
    se_ddi = calculate_standard_error(ddi, n)
    ci_ddi = calculate_ci(ddi, se_ddi)
    return ddi, ci_ddi


# 计算NRI和IDI及其置信区间
def compute_nri_idi_ci(y_test_df, baseline_scores, new_scores, n):
    """
    计算NRI和IDI及其置信区间
    """
    nri, idi = calculate_nri_idi(y_test_df, baseline_scores, new_scores)
    (lower_nri, upper_nri), (lower_idi, upper_idi) = nri_idi_ci(nri, idi, n)
    return (nri, (lower_nri, upper_nri)), (idi, (lower_idi, upper_idi))


# 实现动态辨别指数（DDI）
def dynamic_discrimination_index(auc_values, times):
    """
    计算动态辨别指数（DDI）
    """
    time_diffs = np.diff(times)
    weighted_auc = np.sum((auc_values[:-1] + auc_values[1:]) / 2 * time_diffs)
    total_time = times[-1] - times[0]
    ddi = weighted_auc / total_time
    return ddi


# 计算NRI和IDI
def calculate_nri_idi(y_true_df, baseline_scores, new_scores):
    """
    计算NRI和IDI
    """
    # 事件和非事件
    events = y_true_df['DEATH']
    nonevents = ~events

    # 事件的风险评分
    events_baseline_scores = baseline_scores[events]
    events_new_scores = new_scores[events]

    # 非事件的风险评分
    nonevents_baseline_scores = baseline_scores[nonevents]
    nonevents_new_scores = new_scores[nonevents]

    # 事件的重分类情况
    event_reclassified_up = np.sum(events_new_scores > events_baseline_scores) / len(events_baseline_scores) if len(
        events_baseline_scores) > 0 else 0
    event_reclassified_down = np.sum(events_new_scores < events_baseline_scores) / len(events_baseline_scores) if len(
        events_baseline_scores) > 0 else 0

    # 非事件的重分类情况
    nonevent_reclassified_up = np.sum(nonevents_new_scores > nonevents_baseline_scores) / len(
        nonevents_baseline_scores) if len(nonevents_baseline_scores) > 0 else 0
    nonevent_reclassified_down = np.sum(nonevents_new_scores < nonevents_baseline_scores) / len(
        nonevents_baseline_scores) if len(nonevents_baseline_scores) > 0 else 0

    # 计算NRI
    nri = (event_reclassified_up - event_reclassified_down) - (nonevent_reclassified_up - nonevent_reclassified_down)

    # 计算IDI
    idi = (np.mean(events_new_scores) - np.mean(events_baseline_scores)) - (
                np.mean(nonevents_new_scores) - np.mean(nonevents_baseline_scores))

    return nri, idi


# 将y_test转换为DataFrame以便进行索引和布尔运算
y_test_df = pd.DataFrame({'DEATH': [event for event, _ in y_test], 'TIMEDTH': [time for _, time in y_test]})

# 基准模型预测（这里假设一个简单的基准模型，即所有风险评分都相同）
baseline_risk_scores_test = np.full_like(rsf_risk_scores_test, 0.5)

# ========================
# 计算各项指标及其置信区间
# ========================

# 1. 计算常规C-index及其CI
c_index, ci_c_index = compute_c_index_ci(y_test, rsf_risk_scores_test, n_test)
print(f"RSF Conventional C-index: {c_index:.4f} (95% CI: {ci_c_index[0]:.4f} - {ci_c_index[1]:.4f})")

# 2. 计算时间依赖C-index及其CI
time_dependent_c_index, ci_time_dependent_c_index = compute_time_dependent_c_index_ci(y_train, y_test,
                                                                                      rsf_risk_scores_test_normalized,
                                                                                      n_test)
print(
    f"RSF Time-dependent C-index: {time_dependent_c_index:.4f} (95% CI: {ci_time_dependent_c_index[0]:.4f} - {ci_time_dependent_c_index[1]:.4f})")

# 3. 计算Harrell's C-index及其CI
harrell_c_index, ci_harrell_c_index = compute_harrell_c_index_ci(y_test, rsf_risk_scores_test, n_test)
print(
    f"RSF Harrell's C-index: {harrell_c_index:.4f} (95% CI: {ci_harrell_c_index[0]:.4f} - {ci_harrell_c_index[1]:.4f})")

# 4. 计算ROC AUC及其CI
roc_auc, ci_roc_auc = compute_roc_auc_ci(y_test, rsf_risk_scores_test_normalized, n_test)
print(f"RSF ROC AUC: {roc_auc:.4f} (95% CI: {ci_roc_auc[0]:.4f} - {ci_roc_auc[1]:.4f})")

# 5. 计算动态AUC及其CI
test_times = [y[1] for y in y_test]
min_time = np.min(test_times)
max_time = np.max(test_times)
times = np.linspace(min_time + 1, max_time - 1, 10)

print(f"Min time: {min_time}, Max time: {max_time}")
print(f"Times: {times}")

auc_values, mean_auc, ci_auc, ci_mean_auc = compute_dynamic_auc_ci(y_train, y_test, rsf_risk_scores_test_normalized,
                                                                   times, n_test)
print(f"RSF Dynamic AUC at different times: {auc_values}")
print(f"RSF Mean Dynamic AUC: {mean_auc:.4f} (95% CI: {ci_mean_auc[0]:.4f} - {ci_mean_auc[1]:.4f})")

# 绘制动态AUC曲线
plt.figure()
plt.plot(times, auc_values, color='darkorange', lw=2, label=f'Mean AUC = {mean_auc:.2f}')
plt.fill_between(times, [ci[0] for ci in ci_auc], [ci[1] for ci in ci_auc], color='lightorange', alpha=0.5,
                 label='95% CI')
plt.xlabel('Time')
plt.ylabel('AUC')
plt.title('RSF Dynamic AUC over time')
plt.legend(loc='lower right')
plt.show()

# 6. 计算动态辨别指数（DDI）及其CI
rsf_ddi = dynamic_discrimination_index(auc_values, times)
ddi, ci_ddi = compute_ddi_ci(rsf_ddi, n_test)
print(f"RSF Dynamic Discrimination Index (DDI): {ddi:.4f} (95% CI: {ci_ddi[0]:.4f} - {ci_ddi[1]:.4f})")

# 7. 计算NRI和IDI及其CI
(nri, ci_nri), (idi, ci_idi) = compute_nri_idi_ci(y_test_df, baseline_risk_scores_test, rsf_risk_scores_test_normalized,
                                                  n_test)
print(f"RSF Net Reclassification Improvement (NRI): {nri:.4f} (95% CI: {ci_nri[0]:.4f} - {ci_nri[1]:.4f})")
print(f"RSF Integrated Discrimination Improvement (IDI): {idi:.4f} (95% CI: {ci_idi[0]:.4f} - {ci_idi[1]:.4f})")

# ========================
# 绘制ROC曲线
# ========================
fpr, tpr, thresholds = roc_curve([event for event, _ in y_test], rsf_risk_scores_test_normalized)
rsf_roc_auc_plot = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.fill_between(fpr, tpr - 1.96 * np.sqrt((tpr * (1 - tpr)) / n_test),
                 tpr + 1.96 * np.sqrt((tpr * (1 - tpr)) / n_test), color='lightorange', alpha=0.5, label='95% CI')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RSF ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ========================
# 结果总结
# ========================
print("\n=== 模型评估指标及其95%置信区间 ===")
print(f"C-index: {c_index:.4f} (95% CI: {ci_c_index[0]:.4f} - {ci_c_index[1]:.4f})")
print(
    f"Time-dependent C-index: {time_dependent_c_index:.4f} (95% CI: {ci_time_dependent_c_index[0]:.4f} - {ci_time_dependent_c_index[1]:.4f})")
print(f"Harrell's C-index: {harrell_c_index:.4f} (95% CI: {ci_harrell_c_index[0]:.4f} - {ci_harrell_c_index[1]:.4f})")
print(f"ROC AUC: {roc_auc:.4f} (95% CI: {ci_roc_auc[0]:.4f} - {ci_roc_auc[1]:.4f})")
print(f"Mean Dynamic AUC: {mean_auc:.4f} (95% CI: {ci_mean_auc[0]:.4f} - {ci_mean_auc[1]:.4f})")
print(f"Dynamic Discrimination Index (DDI): {ddi:.4f} (95% CI: {ci_ddi[0]:.4f} - {ci_ddi[1]:.4f})")
print(f"NRI: {nri:.4f} (95% CI: {ci_nri[0]:.4f} - {ci_nri[1]:.4f})")
print(f"IDI: {idi:.4f} (95% CI: {ci_idi[0]:.4f} - {ci_idi[1]:.4f})")
