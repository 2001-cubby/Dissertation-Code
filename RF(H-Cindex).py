from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

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


# 构建并训练普通的随机森林模型
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
rf.fit(X_train, y_train['DEATH'])

# 预测概率
rf_probs_train = rf.predict_proba(X_train)[:, 1]
rf_probs_test = rf.predict_proba(X_test)[:, 1]

# 计算 AUC-ROC 分数
rf_auc_train = roc_auc_score(y_train['DEATH'], rf_probs_train)
rf_auc_test = roc_auc_score(y_test['DEATH'], rf_probs_test)

print(f"Random Forest AUC (Train): {rf_auc_train:.4f}")
print(f"Random Forest AUC (Test): {rf_auc_test:.4f}")

# 计算普通随机森林模型的 C-index
rf_c_index = concordance_index(y_test['TIMEDTH'], -rf_probs_test, y_test['DEATH'])
print(f"Random Forest C-index: {rf_c_index:.4f}")


rf_harrell_c_index = concordance_index(y_test['TIMEDTH'], -rf_probs_test, y_test['DEATH'])
print(f"Random Forest Harrell's C-index: {rf_harrell_c_index:.4f}")