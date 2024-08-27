import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt


data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/fram2.csv"
data = pd.read_csv(data_path)
# 确保事件列为布尔类型
data['DEATH'] = data['DEATH'].astype(bool)

# 创建生存数据格式
y = Surv.from_dataframe('DEATH', 'TIMEDTH', data)

# 选择特征和标签
X = data[['AGE', 'SEX', 'SYSBP', 'DIABP', 'TOTCHOL', 'BMI', 'GLUCOSE', 'CURSMOKE', 'BPMEDS']]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sksurv.ensemble import RandomSurvivalForest

# 训练随机森林模型
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X_train, y_train)

# 预测风险评分
risk_scores_train = rsf.predict(X_train)
risk_scores_test = rsf.predict(X_test)

# 获取测试数据的随访时间范围
test_times = [y[1] for y in y_test]  # 提取测试数据的时间
min_time = np.min(test_times)
max_time = np.max(test_times)

# 确保 times 在测试数据的随访时间范围内
times = np.linspace(min_time + 1, max_time - 1, 100)

# 输出调试信息以检查时间点
print(f"Min time: {min_time}, Max time: {max_time}")
print(f"Times: {times}")

# 计算时间依赖的AUC
auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_test, times)

# 输出结果
print(f"Dynamic AUC at different times: {auc}")
print(f"Mean AUC: {mean_auc}")

# 绘制动态AUC曲线
plt.figure()
plt.plot(times, auc, color='darkorange', lw=2, label=f'Mean AUC = {mean_auc:.2f}')
plt.xlabel('Time')
plt.ylabel('AUC')
plt.title('Dynamic AUC over time')
plt.legend(loc='lower right')
plt.show()

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
