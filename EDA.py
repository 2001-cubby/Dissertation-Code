import pandas as pd

# 加载数据集
data_path = "C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/frmgham2.csv"
data = pd.read_csv(data_path)

# 选择关键变量
key_variables = ['AGE', 'SEX', 'SYSBP', 'TOTCHOL', 'CURSMOKE', 'DIABETES']

# 生成描述性统计信息
desc_stats = data[key_variables].describe()

# 显示性别比例
sex_counts = data['SEX'].value_counts(normalize=True) * 100
sex_counts = pd.DataFrame(sex_counts).T
sex_counts.index = ['SEX Distribution (%)']

# 将性别比例合并到描述性统计表中
desc_stats = pd.concat([desc_stats, sex_counts])

# 显示最终的描述性统计表格
print(desc_stats)

# 保存结果为Excel文件
desc_stats.to_excel("Framingham_Data_Description.xlsx")
