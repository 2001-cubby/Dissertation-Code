#cox前2000波动分析
# 检查前2000天的数据
early_data <- processed_data[processed_data$TIME <= 2000, ]

# 事件发生情况
event_counts <- table(early_data$DEATH)
print(event_counts)

# 绘制生存时间分布
hist(early_data$TIME, breaks=50, main="Survival Time Distribution (<=2000 days)", xlab="Time (days)", ylab="Frequency")

# 使用Cox模型分析特征重要性
early_cox_model <- coxph(Surv(TIME, DEATH) ~ SEX + AGE + SYSBP + DIABP + TOTCHOL + CURSMOKE +
                           CIGPDAY + BMI + DIABETES + BPMEDS + HEARTRTE + GLUCOSE +
                           PREVCHD + PREVAP + PREVMI + PREVSTRK + PREVHYP, data=early_data)

summary(early_cox_model)

# 提取特征的回归系数和p值
coef_summary <- summary(early_cox_model)$coefficients
print(coef_summary)

# 使用交叉验证验证Cox模型的稳定性
library(survival)
library(pec)

cv_cox_model <- coxph(Surv(TIME, DEATH) ~ SEX + AGE + SYSBP + DIABP + TOTCHOL + CURSMOKE +
                        CIGPDAY + BMI + DIABETES + BPMEDS + HEARTRTE + GLUCOSE +
                        PREVCHD + PREVAP + PREVMI + PREVSTRK + PREVHYP, data=processed_data, x = TRUE)

cv_results <- pec::cindex(cv_cox_model, formula=Surv(TIME, DEATH) ~ ., data=processed_data, splitMethod="bootcv", B=100)
print(cv_results)


