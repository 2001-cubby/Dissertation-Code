import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.util import Surv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取数据
data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/fram2.csv"
data = pd.read_csv(data_path)
# 确保事件列为布尔类型
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

# 计算时间依赖C指数
c_index = concordance_index_ipcw(y_train, y_test, risk_scores_test_normalized)
print(f"Time-dependent C-index: {c_index}")

# 计算动态AUC
# 获取测试数据的随访时间范围
test_times = [y[1] for y in y_test]  # 提取测试数据的时间
min_time = np.min(test_times)
max_time = np.max(test_times)

# 确保 times 在测试数据的随访时间范围内
times = np.linspace(min_time + 1, max_time - 1, 10)

# 输出调试信息以检查时间点
print(f"Min time: {min_time}, Max time: {max_time}")
print(f"Times: {times}")

# 计算动态AUC
auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_test_normalized, times)
print(f"Dynamic AUC at different times: {auc}")
print(f"Mean AUC: {mean_auc}")

# 绘制动态AUC曲线
plt.figure()
plt.plot(times, auc, color='navy', lw=2, label=f'Mean AUC = {mean_auc:.2f}')
plt.xlabel('Time')
plt.ylabel('AUC')
plt.title('Dynamic AUC over time')
plt.legend(loc='lower right')
plt.show()

# 计算Brier得分
brier_score = brier_score_loss([event for event, _ in y_test], risk_scores_test_normalized)
print(f"Brier Score: {brier_score}")

# 实现动态辨别指数（DDI）
def dynamic_discrimination_index(auc_values, times):
    """计算动态辨别指数（DDI）"""
    time_diffs = np.diff(times)
    weighted_auc = np.sum((auc_values[:-1] + auc_values[1:]) / 2 * time_diffs)
    total_time = times[-1] - times[0]
    ddi = weighted_auc / total_time
    return ddi

ddi = dynamic_discrimination_index(auc, times)
print(f"Dynamic Discrimination Index (DDI): {ddi}")
