library(survival)

data1 <- read.csv("C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/frmgham2.csv")
head(data1)
summary(data1)

data1 <- read.csv("C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/frmgham2.csv")

# 检查缺失值
colSums(is.na(data1))

# 描述性统计
summary(data1)

# 绘制数值变量的直方图
num_cols <- sapply(data1, is.numeric)
hist_data <- data1[, num_cols]
par(mfrow=c(2, 2))  # 设置多图布局
for(i in 1:ncol(hist_data)) {
  hist(hist_data[,i], main=names(hist_data)[i], xlab="", col="lightblue", border="white")
}

# 绘制箱线图查看异常值
par(mfrow=c(2, 2))
for(i in 1:ncol(hist_data)) {
  boxplot(hist_data[,i], main=names(hist_data)[i], col="lightblue")
}

# 计算数值变量之间的相关性
cor_matrix <- cor(hist_data, use="complete.obs")
print(cor_matrix)

# 使用热图可视化相关性
heatmap(cor_matrix, Rowv=NA, Colv=NA, col=heat.colors(256), scale="column", margins=c(5,10))

# 查看分类变量的分布
cat_cols <- sapply(data1, is.factor)
cat_data <- data1[, cat_cols]
par(mfrow=c(2, 2))
for(i in 1:ncol(cat_data)) {
  barplot(table(cat_data[,i]), main=names(cat_data)[i], col="lightblue", border="white")
}
table(data1$SEX)


# 按性别计算每个变量的均值
mean_by_sex <- aggregate(cbind(AGE, SYSBP, TOTCHOL, CURSMOKE, DIABETES) ~ SEX, data = data1, FUN = mean, na.rm = TRUE)
print(mean_by_sex)

# 计算每个变量的性别比（男性/女性）
sex_ratio <- function(male_value, female_value) {
  return(male_value / female_value)
}

# 计算性别比
sex_ratios <- apply(mean_by_sex[, 2:6], 2, function(x) sex_ratio(x[1], x[2]))
names(sex_ratios) <- colnames(mean_by_sex)[2:6]
print(sex_ratios)

