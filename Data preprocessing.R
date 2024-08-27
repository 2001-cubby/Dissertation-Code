# 加载必要的库
library(dplyr)

# 读取数据
data <- read.csv("C:/Users/廖孔琛/Desktop/曼大/DISSERTATION/FRAMINGHAM_teaching_2021a/csv/frmgham2.csv")

# 查看缺失值情况
missing_summary <- sapply(data, function(x) sum(is.na(x)))
print(missing_summary)

missing_percentage <- sapply(processed_data, function(x) mean(is.na(x)) * 100)
print(missing_percentage)

# 设置缺失值处理的阈值
threshold <- 20

# 创建一个新的数据框来存储处理后的数据
processed_data <- data

# 循环处理每个变量
for (variable in names(data)) {
  missing_pct <- mean(is.na(data[[variable]])) * 100
  
  if (missing_pct > threshold) {
    # 如果缺失值比例大于阈值，删除该变量
    processed_data[[variable]] <- NULL
    cat(variable, "has been removed due to high missing percentage:", missing_pct, "%\n")
  } else if (missing_pct > 0) {
    # 如果缺失值比例小于或等于阈值，进行插补
    if (is.numeric(data[[variable]])) {
      # 数值型变量使用均值插补
      processed_data[[variable]][is.na(data[[variable]])] <- mean(data[[variable]], na.rm = TRUE)
    } else {
      # 非数值型变量使用众数插补
      mode_value <- names(sort(table(data[[variable]]), decreasing = TRUE))[1]
      processed_data[[variable]][is.na(data[[variable]])] <- mode_value
    }
    cat(variable, "has been imputed due to missing percentage:", missing_pct, "%\n")
  }
}

# 查看处理后的数据集
summary(processed_data)
# 将DATA导出为CSV文件
write.csv(processed_data, file = "fram2.csv", row.names = FALSE)



library(randomForestSRC)
library(survival)

processed_data$DEATH <- as.numeric( processed_data$DEATH)
processed_data$TIMEDTH <- as.numeric( processed_data$TIMEDTH)
processed_data$CVD <- as.numeric( processed_data$CVD)
processed_data$TIMECVD <- as.numeric( processed_data$TIMECVD)
surv_object <- Surv(time =  processed_data$TIMEDTH, event =  processed_data$DEATH)

# 设置Cox模型公式
formula_cox <- as.formula(Surv(TIME, DEATH) ~ SEX + AGE + SYSBP + DIABP + TOTCHOL + CURSMOKE +
                            CIGPDAY + BMI + DIABETES + BPMEDS + HEARTRTE + GLUCOSE +
                            PREVCHD + PREVAP + PREVMI + PREVSTRK + PREVHYP)

# 建立Cox回归模型
cox_model <- coxph(formula_cox, data = processed_data)
pred_risk_cox <- predict(cox_model, type = "risk")
# 查看Cox模型结果
summary(cox_model)

# 计算Cox模型的C-index
cindex_cox <- summary(cox_model)$concordance[1]
print(paste("Cox Model C-index:", cindex_cox))

# 提取事件时间和事件状态
event_times <- processed_data$TIME
event_observed <- processed_data$DEATH

# 获取风险评分
pred_risk_cox <- predict(cox_model, type = "risk")

# 手动计算常规C-index
n_pairs <- 0
n_concordant_pairs <- 0

for (i in 1:(length(event_times) - 1)) {
  for (j in (i + 1):length(event_times)) {
    # 只比较两个都未截尾的数据
    if (event_observed[i] == 1 && event_observed[j] == 1) {
      n_pairs <- n_pairs + 1
      # 检查一致性
      if ((pred_risk_cox[i] < pred_risk_cox[j] && event_times[i] < event_times[j]) ||
          (pred_risk_cox[i] > pred_risk_cox[j] && event_times[i] > event_times[j])) {
        n_concordant_pairs <- n_concordant_pairs + 1
      }
    }
  }
}

# 计算常规C-index
cindex_conventional <- n_concordant_pairs / n_pairs
print(paste("Conventional C-index:", cindex_conventional))


# 设置RSF模型公式
formula_rsf <- as.formula(Surv(TIME, DEATH) ~ SEX + AGE + SYSBP + DIABP + TOTCHOL + CURSMOKE +
                            CIGPDAY + BMI + DIABETES + BPMEDS + HEARTRTE + GLUCOSE +
                            PREVCHD + PREVAP + PREVMI + PREVSTRK + PREVHYP)

# 建立RSF模型
rsf_model <- rfsrc(formula_rsf, data = processed_data, ntree = 1000, importance = TRUE)
# 查看RSF模型结果
print(rsf_model)
# 计算变量重要性
importance_rsf <- rsf_model$importance
print(importance_rsf)




# 计算RSF模型的C-index
cindex_rsf <- 1 - rsf_model$err.rate
print(paste("RSF Model C-index:", cindex_rsf))

# 提取RSF模型的风险评分
risk_score_rsf <- rsf_model$predicted.oob

# 普通ROC和AUC for RSF
roc_rsf <- roc(processed_data$DEATH, risk_score_rsf)
plot(roc_rsf, col = "blue", main = "RSF ROC Curve")
auc_rsf <- auc(roc_rsf)
print(paste("RSF Model AUC:", auc_rsf))

# 普通ROC和AUC for Cox
roc_cox <- roc(processed_data$DEATH, pred_risk_cox)
plot(roc_cox, col = "red", main = "Cox ROC Curve")
auc_cox <- auc(roc_cox)
print(paste("Cox Model AUC:", auc_cox))


# 设置AFT模型公式（假设使用对数正态分布）
formula_aft <- as.formula(Surv(TIME1, DEATH) ~ SEX + AGE + SYSBP + DIABP + TOTCHOL + CURSMOKE +
                            CIGPDAY + BMI + DIABETES + BPMEDS + HEARTRTE + GLUCOSE +
                            PREVCHD + PREVAP + PREVMI + PREVSTRK + PREVHYP )

# 建立AFT模型
aft_model <- survreg(formula_aft, data = processed_data, dist = "lognormal")

# 提取AFT模型的预测值（生存时间的对数）
pred_risk_aft <- predict(aft_model, type = "response")

summary(processed_data$TIME)
processed_data$TIME1 <- ifelse(processed_data$TIME <= 0, 0.1, processed_data$TIME)

roc_aft <- roc(processed_data$DEATH, pred_risk_aft)
plot(roc_aft, col = "green", main = "AFT ROC Curve")
auc_aft <- auc(roc_aft)
print(paste("AFT Model AUC:", auc_aft))

library(randomForestSRC)
library(survival)
library(pec)
library(risksetROC)
library(survIDINRI)
library(timeROC)
library(survIDINRI)
library(pROC)
# 提取总体错误率并计算C-index
overall_err_rate <- rsf_model$err.rate[length(rsf_model$err.rate)]
# 计算C-index
cindex_rsf <- 1 - overall_err_rate
print(cindex_rsf)



#HARRELL
# Harrell's C-index for RSF
cindex_rsf <- 1 - rsf_model$err.rate[length(rsf_model$err.rate)]
print(paste("RSF Model Harrell's C-index:", cindex_rsf))

# Harrell's C-index for Cox
cindex_cox <- summary(cox_model)$concordance[1]
print(paste("Cox Model Harrell's C-index:", cindex_cox))

# Harrell's C-index for AFT
aft_surv <- survfit(Surv(TIME, DEATH) ~ 1, data = processed_data, newdata = processed_data, type = "aalen")
cindex_aft <- summary(aft_surv)$concordance[1]
print(paste("AFT Model Harrell's C-index:", cindex_aft))

# Harrell's C-index for AFT
cindex_aft <- survConcordance(Surv(TIME, DEATH) ~ pred_risk_aft, data = processed_data)$concordance
print(paste("AFT Model Harrell's C-index:", cindex_aft))

#计算时间依赖性的ROC曲线
time_roc_rsf <- timeROC(T=processed_data$TIMEDTH, delta=processed_data$DEATH, marker=risk_score_rsf, cause=1, times=seq(0, max(processed_data$TIMEDTH), by=365))
# 绘制时间依赖性的ROC曲线
plot(time_roc_rsf$times, time_roc_rsf$AUC, type="l", xlab="Time (days)", ylab="AUC", main="AFT模型时间依赖性ROC曲线")
# Cox模型的时间依赖性ROC曲线和AUC
time_roc_cox <- timeROC(T=data$TIMEDTH, delta=data$DEATH, marker=pred_risk_cox, cause=1, times=seq(0, max(data$TIMEDTH), by=365))
plot(time_roc_cox$times, time_roc_cox$AUC, type="l", xlab="Time (days)", ylab="AUC", main="COX模型时间依赖性ROC曲线")



# Time-dependent ROC Curve and AUC for RSF
roc_rsf_td <- timeROC(T=processed_data$TIME, delta=processed_data$DEATH, marker=risk_score_rsf, cause=1, times=seq(0, max(processed_data$TIME), by=100), iid=TRUE)
plot(roc_rsf_td, time=2000, col="blue", xlab="False Positive Rate", ylab="True Positive Rate", main="RSF Time-dependent ROC Curve")
auc_rsf_td <- roc_rsf_td$AUC[roc_rsf_td$times == 2000]
print(paste("RSF Model Time-dependent AUC at 2000 days:", auc_rsf_td))

# Time-dependent ROC Curve and AUC for Cox
roc_cox_td <- timeROC(T=processed_data$TIME, delta=processed_data$DEATH, marker=pred_risk_cox, cause=1, times=seq(0, max(processed_data$TIME), by=100), iid=TRUE)
plot(roc_cox_td, time=2000, col="red", xlab="False Positive Rate", ylab="True Positive Rate", main="Cox Time-dependent ROC Curve")
auc_cox_td <- roc_cox_td$AUC[roc_cox_td$times == 2000]
print(paste("Cox Model Time-dependent AUC at 2000 days:", auc_cox_td))


# NRI and IDI for RSF vs Cox
nri_idi_results <- IDI.INRI(TIME = processed_data$TIME, event = processed_data$DEATH, model1 = risk_score_cox, model2 = rsf_model$predicted.oob, cutoff = 1)
print(nri_idi_results)


# Dynamic Discrimination Index for RSF
ddi_rsf <- dynDIS(Surv(TIME, DEATH) ~ rsf_model$predicted.oob, data = processed_data)
print(paste("RSF Model Dynamic Discrimination Index:", ddi_rsf$ddi))

# Dynamic Discrimination Index for Cox
ddi_cox <- dynDIS(Surv(TIME, DEATH) ~ risk_score_cox, data = processed_data)
print(paste("Cox Model Dynamic Discrimination Index:", ddi_cox$ddi))



