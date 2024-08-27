import pandas as pd

data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/fram2.csv"
data = pd.read_csv(data_path)
data['DEATH'] = data['DEATH'].astype(bool)
import matplotlib.pyplot as plt
import seaborn as sns

# 绘制死亡事件发生频率的条形图
plt.figure(figsize=(8, 6))
sns.countplot(x='DEATH', data=data)
plt.title('Frequency of Death Events')
plt.xlabel('Death Event')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['No Death', 'Death'])
plt.show()
# 筛选出死亡事件的数据
death_data = data[data['DEATH'] == True]
import matplotlib.pyplot as plt
import seaborn as sns

# 假设时间列为 'TIME'
plt.figure(figsize=(10, 6))
sns.histplot(death_data['TIME'], bins=30, kde=False)
plt.title('Histogram of Death Events Over Time')
plt.xlabel('Time')
plt.ylabel('Frequency of Death Events')
plt.show()
# 绘制死亡事件时间分布的核密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(death_data['TIME'], shade=True)
plt.title('Density Distribution of Death Events Over Time')
plt.xlabel('Time')
plt.ylabel('Density')
plt.show()
